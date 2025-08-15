import os
import sys
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import socket
import torch
import torch.distributed as dist
import intel_extension_for_pytorch as ipex 
import oneccl_bindings_for_pytorch
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType, MixedPrecision
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
import math
from mpi4py import MPI



# Add current directory to Python path for imports
import sys
sys.path.append('.')

# Import your project modules
from simple_dataloader import PackedDistributedDataLoader
from simple_model import SimpleGPT, Block
from simple_args import args
from simple_trainer import SimpleTrainer

# ---------------- Distributed Setup ----------------
# size = MPI.COMM_WORLD.Get_size()
rank = int(os.environ["PMIX_RANK"])
local_rank = int(os.environ["PALS_LOCAL_RANKID"])
world_size = int(os.environ["WORLD_SIZE"])

os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world_size)
os.environ["LOCAL_RANK"] = str(local_rank)




# local_size = int(os.environ["NGPU"])


if rank == 0:
    master_addr = socket.gethostname()
    print(args)
else:
    master_addr = None
master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)


# assert torch.cuda.is_available()
os.environ["MASTER_ADDR"] = master_addr
os.environ["MASTER_PORT"] = "29500"   # or another open port



dist.init_process_group(
    backend="ccl",
    init_method="env://",
    world_size=world_size,
    rank=rank,
)


# dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])


torch.xpu.set_device(local_rank)

device = "xpu:{}".format(local_rank)
torch.xpu.set_device(device)

master_process = (ddp_rank == 0)
if rank == 0:
    print('world size: ', ddp_world_size)

# Set random seeds
import random
import numpy as np

seed = 133701359 + ddp_rank  # make sure each rank is slightly different
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# ---------------- DataLoader Setup ----------------
# assert args.batch_size % args.ddp_ranks == 0, "ddp ranks does not divide batch size"
B, T = args.device_batch_size, args.sequence_length
# assert args.val_tokens % (B * T * ddp_world_size) == 0




train_loader = PackedDistributedDataLoader(
    args.input_bin, B, T, ddp_rank, ddp_world_size,
    # max_seq_len=args.max_seq_len, min_seq_len=args.min_seq_len,
    # document_separator_token=args.document_separator_token
)
val_loader = PackedDistributedDataLoader(
    args.input_val_bin, B, T, ddp_rank, ddp_world_size,
    # max_seq_len=args.max_seq_len, min_seq_len=args.min_seq_len,
    # document_separator_token=args.document_separator_token
)
if master_process:
    print(f"Training tokens: {train_loader.ntok_total} from {len(train_loader.files)} files")
    print(f"Validation tokens: {val_loader.ntok_total} from {len(val_loader.files)} files")

# ---------------- Model Setup ----------------
# Create a simple config object for the model
# class ModelConfig:
#     def __init__(self, args):
#         self.vocab_size = args.vocab_size
#         self.n_layers = args.n_layers
#         self.model_dim = args.model_dim
#         self.compile = args.compile

# config = ModelConfig(args)
model = SimpleGPT(args).to(torch.bfloat16)
model = model.to(device)

# Count parameters
num_params = sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)
if master_process:
    print(f"Number of non embedding parameters: {num_params:,}")

# Do not compile on XPU for now
# if args.compile:
#     if master_process:
#         print("Compiling model with torch.compile...")
#     model = torch.compile(model)

# Mixed precision policy optimized for throughput
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# FSDP wrapping policy for transformer blocks
transformer_auto_wrapper_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})

# Separate parameters for weight decay
decay = []
no_decay = []
for name, param in model.named_parameters():
    if 'embed' in name:
        no_decay.append(param)
    else:
        decay.append(param)


# Wrap compiled model with FSDP using HYBRID_SHARD strategy (Zero-3) with optimizations
model = FSDP(model,
             mixed_precision=mixed_precision_policy,
             use_orig_params=True,  # Always True when using torch.compile
             auto_wrap_policy=transformer_auto_wrapper_policy,
             sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2,
             forward_prefetch=True,  # Prefetch next layer during forward pass
             limit_all_gathers=True,  # Limit all-gather operations for better performance
             device_id = device
             )

# Optimizer setup with mu-P learning rate adjustment
# effective_learning_rate = args.learning_rate / args.model_dim

effective_learning_rate = args.learning_rate
optimizer = torch.optim.AdamW([
    {'params': decay, 'weight_decay': args.weight_decay}, 
    {'params': no_decay, 'weight_decay': 0.0}
], lr=effective_learning_rate, betas=(0.9, 0.95), fused=True)

current_step = 0  # assume we are starting a new training run

# Learning rate scheduler
def get_cosine_lr(it):
    if it < args.warmup_iters:
        return (it + 1) / args.warmup_iters
    else:
        # Cosine decay: scales from 1 down to 0
        progress = (it - args.warmup_iters) / (args.num_iterations - args.warmup_iters)
        return 0.5 * (1 + math.cos(math.pi * progress))

def wsd_lr_scheduler(it):
    if it < args.warmup_iters:
        return (it + 1) / args.warmup_iters
    elif it > args.num_iterations:
        return 1.0 - (it - args.num_iterations) / (args.num_iterations + 1000)
    else:
        return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, wsd_lr_scheduler)

# Wandb setup with error handling
if master_process:
    try:
        import wandb
        wandb.login()
        wandb_run = wandb.init(project=args.wandb_project, group=args.wandb_group,
                               name=args.wandb_name, id=args.wandb_name, resume='allow', config=args)
        wandb_run.log({'num_params': num_params})
        print(f"Wandb initialized successfully for run: {args.wandb_name}")
    except Exception as e:
        print(f"Wandb initialization failed: {e}")
        print("Continuing training without wandb logging...")
        wandb_run = None
else:
    wandb_run = None

# Checkpoint loading
checkpoint_dir   = f'training_runs/{args.wandb_name}'

# find all "step_*.pt" files
has_prev_checkpoint = os.path.exists(f'{checkpoint_dir}/model_checkpoints') and (len(os.listdir(f'{checkpoint_dir}/model_checkpoints')) > 0)
dist.barrier()
if has_prev_checkpoint:
    model_ckpt_dir   = os.path.join(checkpoint_dir, 'model_checkpoints')
    ckpt_files = [f for f in os.listdir(model_ckpt_dir)
                if f.startswith("step_") and f.endswith(".pt")]

    
    # 1) pick the latest step
    load_ckpt_step = max(
        int(f.split("_")[1].split(".")[0])
        for f in ckpt_files
    )
    ckpt_path = os.path.join(model_ckpt_dir, f"step_{load_ckpt_step}.pt")
    print(f"LOADING checkpoint step {load_ckpt_step}")

    # 2) rank 0 loads; others set up a placeholder
    obj = [None]
    if ddp_rank == 0:
        obj[0] = torch.load(ckpt_path, map_location="cpu")

    # 3) broadcast the Python dict to every rank
    dist.broadcast_object_list(obj, src=0)
    ckpt = obj[0]

    # 4) load model + optimizer + scheduler under FULL_STATE_DICT
    FSDP.set_state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(rank0_only=True),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optim_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    # 5) restore each rank's dataloader state
    loader_states = ckpt['train_loader_state']
    _, shard, pos = loader_states[ddp_rank].tolist()
    train_loader.load_from_ckpt(int(shard), int(pos))

    current_step = ckpt['step']
dist.barrier()

# Create trainer and start training
trainer = SimpleTrainer(args, model, optimizer, scheduler,
                        train_loader, val_loader, wandb_run, args.wandb_name, master_process, device, 
                        ddp_world_size, ddp_rank, current_step)

trainer.train()

# if master_process:
    # print(f"Peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
print(f"Cleaning up on rank {rank}")
dist.destroy_process_group() 
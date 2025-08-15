#!/bin/bash -l
#PBS -l walltime=0:20:00
#PBS -l select=40
#PBS -N 40_node_test_2
#PBS -k doe
#PBS -j oe
#PBS -A AuroraGPT
#PBS -l filesystems=home:flare
#PBS -q prod
#PBS -M swheeler@anl.gov                                                  
#PBS -m bae 






# proxy settings
if [[ ! "${HOSTNAME}" =~ aurora-uan ]]; then
  export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
  export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
  export http_proxy="http://proxy.alcf.anl.gov:3128"
  export https_proxy="http://proxy.alcf.anl.gov:3128"
  export ftp_proxy="http://proxy.alcf.anl.gov:3128"
  export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"
fi



export CCL_ALLREDUCE=topo
export CCL_ALLREDUCE_SCALEOUT=direct

export CCL_ALLGATHER=topo
export CCL_ALLGATHERV=topo
export CCL_ALLGATHER_SCALEOUT=direct
export CCL_ALLGATHERV_SCALEOUT=direct
export CCL_ALLGATHERV_MEDIUM_SIZE_THRESHOLD=0

export CCL_REDUCE_SCATTER=topo
export CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL=1
export CCL_REDUCE_SCATTER_SCALEOUT=direct
export CCL_ENABLE_SYCL_KERNELS=0

module load frameworks

export CCL_WORKER_AFFINITY=1,9,17,25,33,41,53,61,69,77,85,93
export CPU_BIND="list:2-8:10-16:18-24:26-32:34-40:42-48:54-60:62-68:70-76:78-84:86-92:94-100"
export NUMEXPR_MAX_THREADS=7
export OMP_NUM_THREADS=7


export NGPU=12
export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | wc -l)
export NNODES=$PBS_JOBSIZE
export WORLD_SIZE=$((NGPU*PBS_JOBSIZE))

export MASTER_ADDR=$(head -n 1 ${PBS_NODEFILE})

min=16384
max=32768
random_number=$(( $RANDOM % (max - min + 1) + min ))
export MASTER_PORT=$random_number


# calculate total ranks
NTOTRANKS=$(( NNODES * NGPU ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NGPU}"

cd $PBS_O_WORKDIR
echo $(pwd)
source training_venv/bin/activate

# Run 100M parameter model training with optimized settings for 1-hour jobs
# Disable wandb if there are issues, but continue training
mpiexec -n ${NTOTRANKS} --ppn ${NGPU} python -u model_training_scripts/simple_execute.py \
    --config model_training_scripts/configs/dense_100m.yaml \
    --compile \
    --save_every=500 \
    --val_loss_every=50 \
    --wandb_name="dense_40_node_time_test" >> dense_40_node_time_test.txt 2>&1 
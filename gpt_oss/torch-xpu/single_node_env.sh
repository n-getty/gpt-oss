export http_proxy="proxy.alcf.anl.gov:3128"
export https_proxy="proxy.alcf.anl.gov:3128"
module load frameworks
#conda create -n gpt-oss-env-312-clean python==3.12.0 -y
#pip install -r requirements-alt.txt
conda activate gpt-oss-env-312-clean

# Check torch/ipex compatability
python -c "import torch; import intel_extension_for_pytorch as ipex; print('✅ Success! IPEX version:', ipex.__version__)"

# Verify XPUs visible
sycl-ls

pip install -e ".[test]"

# CCL Exports
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_ALLREDUCE_SCALEOUT="direct:0-1048576;rabenseifner:1048577-max"
export CCL_BCAST=double_tree
export CCL_KVS_MODE=mpi
export CCL_KVS_USE_MPI_RANKS=1

# Run the torch based generate code
mpiexec -n 6 python -m gpt_oss.generate /flare/AuroraGPT/model-weights/hub/models--openai--gpt-oss-120b/snapshots/4c0fa49a43c232dc2a1cae033ae1fb7961cf8220/original/

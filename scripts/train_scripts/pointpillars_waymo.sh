#!/bin/bash
# Launch this script on each training node, for e.g. with SLURM
# Scale number of CPUs and memory according to number of GPUs
#SBATCH --job-name=objdet_4
#SBATCH --gres=gpu:4
#SBATCH -c 48
#SBATCH --mem=384G
#SBATCH --output="./slurm-%x-%j-%N.log"

if [ "$#" -ne 2 ]; then
    echo "Please, provide the training framework: torch/tf and dataset path."
    exit 1
fi

# Use launch node as master, if not set
export MASTER_ADDR=${MASTER_ADDR:-$SLURMD_NODENAME}
export MASTER_PORT=${MASTER_PORT:-29500}
# Use all available GPUs, if not set. Must be the same for ALL nodes.
export DEVICE_IDS=${DEVICE_IDS:-$(nvidia-smi --list-gpus | cut -f2 -d' ' | tr ':\n' ' ')}
export NODE_RANK=${NODE_RANK:-SLURM_NODEID} # Pass name of env var

echo Started at: $(date)
pushd ../..
# Launch on each node:
srun -l python scripts/run_pipeline.py "$1" -c ml3d/configs/pointpillars_waymo.yml \
    --dataset_path "$2" --pipeline ObjectDetection \
    --pipeline.num_workers 0 --pipeline.pin_memory False \
    --pipeline.batch_size 4 --device_ids $DEVICE_IDS \
    --backend nccl \
    --nodes $SLURM_JOB_NUM_NODES \
    --node_rank "$NODE_RANK" \
    --host "$MASTER_ADDR" --port "$MASTER_PORT"

echo Completed at: $(date)
popd

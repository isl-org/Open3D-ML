#!/bin/bash
# Launch this script on each training node, for e.g. with SLURM. This example
# launches 8 workers total: SLURM launches this on 2 nodes and run_pipline.py
# launches 4 workers on each node.
#SBATCH --job-name=objdet_2x4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48      # scale by n_gpus_per_node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=384G              # scale by n_gpus_per_node
#SBATCH --output="./slurm-%x-%j-%N.log"

if [ "$#" -ne 2 ]; then
    echo "Please, provide the training framework: torch/tf and dataset path."
    exit 1
fi

# Use launch node as main, if not set
export PRIMARY_ADDR=${PRIMARY_ADDR:-$SLURMD_NODENAME}
export PRIMARY_PORT=${PRIMARY_PORT:-29500}
# Use all available GPUs, if not set. Must be the same for ALL nodes.
export DEVICE_IDS=${DEVICE_IDS:-$(nvidia-smi --list-gpus | cut -f2 -d' ' | tr ':\n' ' ')}
export NODE_RANK=${NODE_RANK:-SLURM_NODEID} # Pass name of env var

echo Started at: $(date)
pushd ../..
# Launch on each node
srun -l python scripts/run_pipeline.py "$1" -c ml3d/configs/pointpillars_waymo.yml \
    --dataset_path "$2" --pipeline ObjectDetection \
    --pipeline.num_workers 0 --pipeline.pin_memory False \
    --pipeline.batch_size 4 --device_ids $DEVICE_IDS \
    --backend nccl \
    --nodes $SLURM_JOB_NUM_NODES \
    --node_rank "$NODE_RANK" \
    --host "$PRIMARY_ADDR" --port "$PRIMARY_PORT"

echo Completed at: $(date)
popd

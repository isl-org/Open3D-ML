#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 14
#SBATCH --gres=gpu:1

function term_handler()
{
    echo "Cleaning up.."
    rm -rf /dev/shm/slurm-${SLURM_JOB_ID}
}

trap term_handler SIGTERM SIGINT ERR EXIT

if [ "$#" -ne 2 ]; then
    echo "Please, provide the the training framework: torch/tf and dataset path"
    exit 1
fi

export OPEN3D_ML_ROOT=$HOME/Documents/Open3D/Code/Open3D-ML

# pushd ../..
# dlprof --mode=pytorch -f true --reports=all --iter_start=10 \
# nsys profile -f true -o pointpillars --export sqlite \
#     python scripts/run_pipeline.py "$1" -c ml3d/configs/pointpillars_scannet.yml \
#     --dataset_path "$2"
# echo Training done. Now parsing / analysing ...
# python -m pyprof.parse pointpillars.sqlite > pointpillars.dict
# python -m pyprof.prof --csv pointpillars.dict > pyprof-pointpillars.csv
# Cache up to 8G data in RAM, larger data in local disk
if (( $(du -m "$2" | cut -f1) > 8192 )); then
    LOCALDIR="$TMPDIR"/data
else
    LOCALDIR=/dev/shm/slurm-${SLURM_JOB_ID}/data
fi
mkdir -p "$LOCALDIR"
echo Copying training data to $LOCALDIR
# rsync -ah --info=progress2 --include='*_000100_*' --include='*/' --exclude='*' "$2/." "$LOCALDIR"
rsync -ah --info=progress2 "$2/." "$LOCALDIR"
LOCALDIR="$2"
tensorboard --logdir /mnt/beegfs/tier1/vcl-nfs-work/ssheorey/Open3D/scannet-frames/train_logs \
  --bind_all --port=6036 &
echo "Starting training..."
python scripts/run_pipeline.py "$1" -c ml3d/configs/pointpillars_scannet_frames.yml \
    --dataset_path "$LOCALDIR" \
    --cache_dir "$LOCALDIR/cache" \
    --main_log_dir /mnt/beegfs/tier1/vcl-nfs-work/ssheorey/Open3D/scannet-frames/logs-$(date -I) \
    --pipeline.train_sum_dir /mnt/beegfs/tier1/vcl-nfs-work/ssheorey/Open3D/scannet-frames/train_logs \
    --pipeline ObjectDetection
    #--ckpt_path /mnt/beegfs/tier1/vcl-nfs-work/ssheorey/Open3D/logs/PointPillars_Scannet_torch/checkpoint/ckpt_00030.pth \
# popd

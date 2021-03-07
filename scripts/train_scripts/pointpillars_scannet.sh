#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --gres=gpu:1

if [ "$#" -ne 2 ]; then
    echo "Please, provide the the training framework: torch/tf and dataset path"
    exit 1
fi

pushd ../..
cp -r "$2" "$TMPDIR"
python scripts/run_pipeline.py "$1" -c ml3d/configs/pointpillars_scannet.yml \
    --dataset_path "$TMPDIR" --pipeline ObjectDetection
popd

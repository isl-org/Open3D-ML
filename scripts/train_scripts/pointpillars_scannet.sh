#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH --gres=gpu:1

if [ "$#" -ne 2 ]; then
    echo "Please, provide the the training framework: torch/tf and dataset path"
    exit 1
fi

pushd ../..
# dlprof --mode=pytorch -f true --reports=all --iter_start=10 \
nsys profile -f true -o pointpillars --export sqlite \
    python scripts/run_pipeline.py "$1" -c ml3d/configs/pointpillars_scannet.yml \
    --dataset_path "$2"
echo Training done. Now parsing / analysing ...
python -m pyprof.parse pointpillars.sqlite > pointpillars.dict
python -m pyprof.prof --csv pointpillars.dict > pyprof-pointpillars.csv
popd

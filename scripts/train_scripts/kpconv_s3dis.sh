#!/bin/bash
#SBATCH -p gpu 
#SBATCH --gres=gpu:1 

if [ "$#" -ne 1 ]; then
    echo "Please, provide the the training framework: torch/tf."
    exit 1
fi

cd ../..
python scripts/semseg.py $1 -c ml3d/configs/kpconv_s3dis.yml \
--dataset_path /export/share/Datasets/S3DIS

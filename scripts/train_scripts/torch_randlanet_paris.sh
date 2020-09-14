#!/bin/bash
#SBATCH -p gpu 
#SBATCH --gres=gpu:1 
cd ../Open3D-ML
python scripts/semseg.py torch -c ml3d/configs/randlanet_parislille3d.yml --dataset_path /export/share/Datasets/Paris_Lille3D

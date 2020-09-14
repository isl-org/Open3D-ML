python scripts/semseg.py tf -c ml3d/configs/randlanet_toronto3d.yml --dataset_path ../dataset/Toronto3D

python scripts/semseg.py torch -c ml3d/configs/randlanet_semantickitti.yml --dataset_path ../dataset/SemanticKITTI

python scripts/semseg.py torch -c ml3d/configs/randlanet_toronto3d.yml --dataset_path ../dataset/Toronto3D

python scripts/semseg.py torch -c ml3d/configs/kpconv_semantickitti.yml --dataset_path ../dataset/SemanticKITTI

python scripts/semseg.py torch -c ml3d/configs/kpconv_toronto3d.yml --dataset_path ../dataset/Toronto3D

python scripts/semseg.py torch -c ml3d/configs/kpconv_s3dis.yml --dataset_path ../dataset/S3DIS

python scripts/semseg.py torch -c ml3d/configs/kpconv_parislille3d.yml --dataset_path ../dataset/Paris_Lille3D

python scripts/semseg.py torch -c ml3d/configs/randlanet_parislille3d.yml --dataset_path ../dataset/Paris_Lille3D

python scripts/semseg.py torch -c ml3d/configs/randlanet_semantic3d.yml --dataset_path ../dataset/Semantic3D


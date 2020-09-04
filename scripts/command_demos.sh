# 
python scripts/run.py torch -m RandLANet \
-d SemanticKITTI --cfg_dataset ml3d/configs/default_cfgs/semantickitti.yml --dataset_path ../dataset/SemanticKITTI 

# test config file
python scripts/run.py torch -c ml3d/configs/kpconv_semantickitti.yml \
--pipeline.batch_size 2

# Copyright (c) 2022 by Andreas Albrecht. All rights reserved.

#!/usr/bin/env bash

# Run this bash script in experiments folder within the docker environment project-dev to export a tf model as frozen graph
# bash -i export_model.sh

# Specify experiment folder
experiment="experiment_1_00"

# Model configuration filepath
pipeline_config_path="/app/project/experiments/$experiment/pipeline_$experiment.config"
echo "pipeline_config_path: $pipeline_config_path"

# Model checkpoint directory
model_checkpoint_dir="/app/project/experiments/$experiment/model_checkpoints"
echo "model_checkpoint_dir: $model_checkpoint_dir"

# Create model export directory if it does not exist yet
model_export_dir="/app/project/experiments/$experiment/exported_model"
echo "model_export_dir: $model_export_dir"
mkdir -p ${model_export_dir}

# Export the trained model from last available checkpoint to the model export directory
python exporter_main_v2.py --input_type image_tensor \
  --pipeline_config_path $pipeline_config_path \
  --trained_checkpoint_dir $model_checkpoint_dir \
  --output_directory $model_export_dir


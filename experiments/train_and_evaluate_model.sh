# Copyright (c) 2022 by Andreas Albrecht. All rights reserved.

#!/usr/bin/env bash

# Run this bash script in project folder within the docker environment project-dev to train a new tf model
# bash -i export_model.sh

# Specify experiment folder
experiment="experiment_1_00"

# Model configuration filepath
pipeline_config_path="/app/project/experiments/$experiment/pipeline_$experiment.config"
echo "pipeline_config_path: $pipeline_config_path"

# Model checkpoint directory
model_checkpoint_dir="/app/project/experiments/$experiment/model_checkpoints"
echo "model_dir: $model_checkpoint_dir"
echo "model_checkpoint_dir: $model_checkpoint_dir"

# Train the tf model as specified by the pipeline configuration
python model_main_tf2.py \
    --pipeline_config_path=$pipeline_config_path \
    --model_dir=$model_checkpoint_dir
#python model_main_tf2.py \
#    --pipeline_config_path=$pipeline_config_path \
#    --model_dir=$model_checkpoint_dir \
#    --num_train_steps=25000
#    --eval_training_data=True
#    --sample_1_of_n_eval_on_train_examples=1000
#    --num_eval_steps=25
#    --sample_1_of_n_eval_examples=1

# Evaluate the tf model on the validation set as specified by the pipeline configuration
python model_main_tf2.py \
    --pipeline_config_path=$pipeline_config_path \
    --model_dir=$model_checkpoint_dir \
    --checkpoint_dir=$model_checkpoint_dir
#python model_main_tf2.py \
#    --pipeline_config_path=$pipeline_config_path \
#    --model_dir=$model_checkpoint_dir \
#    --checkpoint_dir=$model_checkpoint_dir \
#    --sample_1_of_n_eval_examples=1


# Copyright (c) 2022 by Andreas Albrecht. All rights reserved.

#!/usr/bin/env bash

# Run this bash script in experiments folder within the docker environment project-dev to perform inference on all train/val/test videos
# bash -i create_inference_videos.sh

# Specify experiment folder
experiment="experiment_1_00"

# Specify data subset
#datasubset="train"
datasubset="val"
#datasubset="test"

# Pipeline configuration path
pipeline_config_path="/app/project/experiments/$experiment/pipeline_$experiment.config"
echo "pipeline_config_path: $pipeline_config_path"

# Exported model directory
model_dir="/app/project/experiments/$experiment/exported_model/saved_model"
echo "model_dir: $model_dir"


# Create target directory if it does not exist yet
target_dir="/app/project/experiments/$experiment/inference/videos/$datasubset"
mkdir -p ${target_dir}

# Source directory where the data sets reside (tfrecord files)
source_dir="/app/project/data/waymo/$datasubset"

# Specify path to the label map
labelmap_path="/app/project/label_map.pbtxt"
echo "labelmap_path: $labelmap_path"

# Loop over all *.tfrecord files in the given source directory and perform inference on data sets
for source_path in `find $source_dir -type f -name "*.tfrecord"`
do
    # Show source filepath
    echo "source_path: $source_path"
    
    # Get source file id (snip off extensions - even double extensions)
    source_file_id=$(basename "$source_path" | cut -d. -f1)
    echo "source_file_id: $source_file_id"
    
    # Set target filepath
    target_file="$source_file_id.mp4"
    target_path="$target_dir/$target_file"
    echo "target_path: $target_path"
    
    # Perform inference on source file
    python inference_video.py \
     	--labelmap_path $labelmap_path \
    	--model_path $model_dir \
    	--tf_record_path $source_path \
    	--config_path $pipeline_config_path \
    	--output_path $target_path
done


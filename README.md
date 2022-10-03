# Object Detection in an Urban Environment

Computer vision project submission 1 of Udacity's "Self-Driving Car Engineer" Nanodegree Program about 2D Object Detection in an Urban Environment using Waymo Open Dataset focusing on exploratory experiments using TensorFlow's Object Detection API. This project has been developed based on the [starter code](https://github.com/udacity/nd013-c1-vision-starter) provided by Udacity. 

## Introduction

Real-world traffic environments a self-driving car (SDC) targets to operates in are quite complex and subject to permanent changes. Such ever changing "open context" environments cannot be captured by a digital map. Therefore, a SDC needs highly sophisticated perception capabilities, e.g. based on vision, radar or lidar sensors, in order to "see" or perceive the world around and react appropriately on any possible situation. The perception functions of a SDC (e.g. like object detection) are all highly safety-critical. A potential failure like overlooking a pedestrian would easily lead to a potentially fatal accident.

This project focuses on exploring some basic capabilities of TensorFlow Object Detection API using object detection from the domain of self-driving cars as an example. We will train a deep neural network (e.g. a resnet50) taken from [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) for object detection. We will use transfer learning to train it on detecting three tpyes of object classes (vehicles, pedestrians and cylcists) using different configuration options of TF Object Detection API and a selected subset from [Waymo Open dataset](https://waymo.com/open/). The data set contains labeled vision and lidar data sequences recorded by some vehicles of the Waymo fleet in US using multiple camera and lidar sensors.

[OPTIONAL] - The files can be downloaded directly from Waymo's website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tfrecords. You may also use a script that comes with this proejct to download the data from its original source (s. Download and process the data).

## Local Setup

### Installation Instructions and Requirements

Before you start please ensure that you have [git](https://github.com/git-guides/install-git) and [git-lfs](https://github.com/git-lfs/git-lfs#getting-started) installed on your system. Clone this repository to a working directory of your choice.
```
sudo apt-get update
sudo apt-get install git
sudo apt-get install git-lfs
cd <working directory> # e.g. cd ~/workspace
git clone https://github.com/AndiA76/SDCND01_2D_Object_Detection.git
```
For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container called `project-dev`and install all prerequisites.

### Run the Docker Container

Run the following command to start the docker container and mount your project folder:
```
docker run --gpus all -v ~/workspace/SDCND01_2D_Object_Detection:/app/project --network=host -it project-dev bash
```
If needed you can increase shared memory size by passing in the `--shm-size=??MB` argument with the command:
```
docker run --shm-size=32GB --gpus all -v ~/workspace/SDCND01_2D_Object_Detection:/app/project --network=host -it project-dev bash
```
If you want to work in parallel in your current docker environment, e.g. while a jupyter notebook or a training process is running in your first terminal, you a second terminal with access to your docker environment. Therefore, open a second terminal and get the container id of you running docker container by the following command: 
```
docker ps
```
It returns amongst others the container id. Please copy it and enter the following command in the second terminal. Please replace `<container_id>` accordingly.
```
docker exec -it <container_id> bash
```
You can now work in parallel and start other processes.

### Run a Jupyter Notebook inside the Docker Container
If jupyter notebook runs inside a docker environment you need to grant root access in order to access jupyter via the browser. From within the docker environment launch jupyter notebook allowing for root access rights using this command:
``` 
jupyter notebook --allow-root
```
Then copy the URL jupyter has returned and paste it into your browser to open the jupyter notebook.

Jupter notebook now blocks your current terminal for its logs. If you want to work in parallel in the same docker environment while jupyter notebook is running in the first terminal you need to open a second terminal and proceed as described above.

## Project Structure

### Folder Structure
If you run this project in a docker container the root folder structure is organized as follows:
```
/app/models - contains the TensorFlow models and tf object detection api
/app/project - contains the project files
/app/protobuf - contains google protobuf files
```
The project directory itself is structured as follows:
```
/app/project/build/ - contains the files to build the docker environment for this project
/app/project/data/ - contains the waymo data after download and processing (empty at the beginning)
/app/project/experiments/ - contains the training scripts and experiments incl. model checkpoints an results (empty at the beginning)
/app/project/thirdparty/ - contains a modified a tutorial how to access and visualize waymo data
/app/project/ - contains the source files
```
### Source Files
```
/app/project/thirdparty/
    - Explore_Raw_TFRecordFile.ipynb - modified tutorial that shows how to access and visualize waymo data
/app/project/experiments/
    - exporter_main_v2.py - script that comes with TF Object Detection API to export a model as a frozen graph
    - export_model.sh - bash script to export a model
    - label_map.pbtxt - label map file covering the classes "vehicle", "pedestrian" and "cyclist"
    - model_main_tf2.py - script that comes with tF Object Detection APIA to launch a training or an evaluation process
    - train_and_evaluate_model.sh - bash script to sequentially run a training and an evaluation process in the same terminal
/app/project/
    - create_inference_videos.sh - bash script to create an inference video
    - create_splits.py - script to split a dataset into train and val subsets or train, val and test subsets by a given ratio
    - download_and_process_tfrecords.py - script to download and process a set of tfrecord files from Waymo's open data set
    - download_tfrecords.py - script to only download the a set of tfrecord files frmo Waymo's open data set
    - edit_config.py - script to help modifying the pipeline.config file
    - Experimental_Results.md - writeup of the transfer learning experimental results
    - Exploratory_Data_Analysis_of_Original_Waymo_Data.ipynb - exploratory data analysis of the original raw data set 
    - Exploratory_Data_Analysis_Part1.ipynb - exploratory data analysis (part 1) of stripped and downsampled training and validation data before splitting
    - Exploratory_Data_Analysis_Part2.ipynb - exploratory data analysis (part 2) comparing the training and validation sub-set after splitting
    - Explore_Augmentations.ipynb - exploratory analysis of data augmentations options of the training pipeline
    - Explore_TF_Default_Augmentations.ipynb - exploratory analysis of augmentation options supported by TF Object Detection API
    - filenames.txt - list of source links to available tfrecord files in the Waymo open dataset
    - filenames_test.txt - list of source links to the tfrecord files selected to be part of the test set
    - filenames_train_and_val.txt - list of source links to the tfrecord files selected to be part of the training and validation set
    - inference_video.py - script to create inference videos using a trained SSD model
    - label_map.pbtxt - label map file covering the classes "vehicle", "pedestrian" and "cyclist"
    - pipeline.config - default pipeline config file for "ssd_resnet50_v1_fpn_keras"
    - process_tfrecords.py - script to strip and downsample a set of tfrecord files using a user-specified downsampling rate
    - README.md - this readme file
    - utils.py - utility functions
```
The bash sripts are meant to ease entering longer commands with many arguments or to run a command / script multiple times. You can modify the bash script according to the command you want to run. Then double-check, save and run the bash script, e.g.:
```
bash -i ./create_inference_videos.sh
```


## Data

### Data Set
After having set up the docker environemnt, you can download e.g. the first 100 tfrecord files from a list of available tfrecord files (s, [filenames.txt](./filenames.txt)) from [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/). Each tfrecord file contains a multi-sensor data sequence recorded at a frame rate of 10 fps. For training an object detection model only the video data from the front camera including the corresponding meta data and ground truth labels is used. We will strip off the rest (e.g. lidar data).

For training and cross-validation we will use 97 tfrecord files. For testing, we will keep 3 tfrecord files that the model will not see during training. These 3 files are defined by Udacity and will not be altered. Assuming that subsequent images in a video sequence are very similar, the data for training and validation is downsampled only using 1 of every 10 images. For testing the video data is used as is without downsampling.

### Data Structure
The data for training, cross-validation and testing will be organized as follows in the project:
```
/app/project/data/waymo/
    - raw - contains the original tfrecord files after download
    - processed - contains the reduced tfrecord files after processing
    - training_and_validation - contains 97 processed files to train and validate your model before splitting for data analysis
    - train: contains the training data after splitting
    - val: contains the cross-validation data after splitting
    - test - contains 3 specific tfrecord files to test your model and create inference videos
```

### Download and Process the Data

For this project, we only need a subset of the data provided (for example, we do not need the Lidar data or the data of the other video cameras besides the front video). Therefore, we are going to download and trim each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo tfrecord file and saves them in the TF Object Detection API format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). The label map specified by the `label_map.pbtxt` file is reduced to only three classes (1: vehicle, 2: pedestrian, 4: cyclist).

For downloading the data you need to have a valid google account and you have register on [Waymo Open dataset](https://waymo.com/open/). If you don't have a google account yet you can create one following this link: [Create Google Account](https://accounts.google.com/signup/v2/webcreateaccount?hl=en&flowName=GlifWebSignIn&flowEntry=SignUp)

In order to be able to download the tfrecord files from [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) you need to authenticate yourself in your running docker environment with your google account. Therefore, enter:
```
gcloud auth login
```
A link will appear that brings to a google page that provides a verification code. Copy this code and paste it to the terminal prompt that says `Enter verification code: `.

Now you can start the data download and processing. For downloading an immediate stripping the tfrecord files you can run the `download_and_process_tfrecords.py` script using the following command:
```
python download_and_process_tfrecords.py \
    --data_dir {data set root dir} \
    --file_list {source path list of files available for download, defaults to filenames.txt} \
    --size {number of files you want to download, defaults to 100} \
    --extract_1_of_N {extract only every Nth frame, defaults to 1} \
    --cleanup {clean up raw tfrecord files after download to save disk space, defaults to True}
```
e.g.
```
python download_and_process_tfrecords.py --data_dir /app/project/data/waymo/ --file_list filenames.txt --size 100 --extract_1_of_N 10 --cleanup False
```
By default, the script downloads the first 100 files from the file list (with the source filepaths to the google cloud bucket) specified by the `--file_list` parameter (defaults to `filenames.txt`) to a `raw` folder, which is created in the data directory given by `--data_dir` (e.g. `--data_dir = ./data/waymo/`). This might take quite a while, so be patient!
The number of files to be downloaded can be changed by the `--size` parameter. If you want to download and process only specific files, e.g. using a user-defined downsampling rate, you can create your own file list. The `--size` parameter is automatically limited to the length of the file list if the latter is not long enough. 
After downloading the raw tfrecord files to the `raw`folder, only the front camera images and labels are extracted and evtl. downsampled. All other data contents are stripped off. By default, no downsampling is applied (`--extract_1_of_N = 1`). If you want to downsample the tfrecord files you can change `--extract_1_of_N` to specify your desired downsampling rate, which is then applied to all tfrecord files from the file list. For instance, `--extract_1_of_N = 10` extracts only every 10th front camera image frame incl. the labels from each tfrecord file.
After processing, the raw tfrecord files are deleted by default in order to save disk space. However, when you strip the raw data you will loose the ability to access the stripped off parts of the data, e.g. including certain meta data like time of day, which may be relevant for analysing and splitting the data set. If you also want to analyse the original data you need to keep the raw tfrecord files after download by setting `--cleanup = False`.
Once the script is done, you can look inside your `<data_dir>/raw` and `<data_dir>/processed` folders to see if the files have been downloaded and processed correctly. For testing if everything works you may want to start with a smaller number of files.

In the following we will download and process the following 3 tfrecord files keeping the original frame rate of 10 fps (without downsampling) and move them to the `test` folder (this setup was given by Udacity):
- `segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord`
- `segment-12012663867578114640_820_000_840_000_with_camera_labels.tfrecord`
- `segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord`

The other 97 tfrecord files from the first 100 in [filenames.txt](./filenames.txt) are downsampled by setting `--extract_1_of_N = 10`. So only every 10th frame will be extracted. Afterwards we move the processed tfrecords from the `processed` folder to the `training_and_validation` folder.

If you want to download the raw tfrecord files without processing you can use `download_tfrecords.py` using the following command.
```
python download_tfrecords.py \
    --data_dir {data set root dir} \
    --file_list {source path list of files available for download, defaults to filenames.txt} \
    --size {number of files you want to download, defaults to 100}
```
e.g.
```
python download_tfrecords.py --data_dir /app/project/data/waymo/ --file_list filenames.txt --size 100
```
You can process the downloaded tfrecord files or a part of them later using `process_tfrecords.py` via the command below. You have to specify the files you want to process in a list. Therefore, you can copy `filenames.txt` and delete the source files you don't need. Don't worry about the paths, the script will just used the base names and check if the corresponding raw tfrecords exist in the source directory.
```
python process_tfrecords.py \
    --data_dir {data set root directory where the processed tfrecord files are stored in a subfoleder 'processed'} \
    --source_dir{source directory where the raw tfrecord files are stored} 
    --file_list {list of specific tfrecord files to be processed, defaults to filenames.txt} \
    --extract_1_of_N {extract only every Nth frame, defaults to 1} \
```
e.g.
```
python process_tfrecords.py --data_dir /app/project/data/waymo/ --source_dir /app/project/data/waymo/raw --file_list filenames_train_and_val.txt --extract_1_of_N 10
python process_tfrecords.py --data_dir /app/project/data/waymo/ --source_dir /app/project/data/waymo/raw --file_list filenames_test.txt --extract_1_of_N 1

```
Hints:
* In my local docker environment, parallel processing using ray sometimes caused cuda errors or could not find some downloaded files and broke. If you can't fix this you may want to use serial processing instead. Please comment out the ray.remote decorator and the parallel procesing part and uncomment the serial process part. It takes longer but it worked stable for me.
* When processing data in the docker environment and writing the output files to the shared memory the output files are usually locked automatically by docker. In order to use them without restrictions outside the docker environment you can change the ownership of the files or sub-directories:

`sudo chown -R username:group <directory or file(s)>` e.g. `sudo chown -R $USER:$USER <dir>` or `sudo chown -R $USER:$USER <dir>/*.*`

### Dataset Analysis
As we loose information when stripping the data we first want to analyse the original Waymo data directly after download and before stripping and downsampling. This also gives a clear overview on the number of availabel images. The following modified third party tutorial gives an overview how to access the raw Waymo data [Explore_Raw_TFRecordFile.ipynb](./third_party/Explore_Raw_TFRecordFile.ipynb). The analysis results of the raw Waymo data can be found in this notebook [Exploratory_Data_Analysis_of_Original_Waymo_Data.ipynb](./Exploratory_Data_Analysis_of_Original_Waymo_Data.ipynb). It also contains a data analysis of some high-level features (e.g. daytime / nighttime) contained in the training, validation and test data sub-sets used in this project.

After stripping and downsampling the raw tfrecords we perform another exporatory data analysis (part 1) on the overall training and validation data before splitting. This analysis can be found in this notebook [Exploratory_Data_Analysis_Part1.ipynb](./Exploratory_Data_Analysis_Part1.ipynb). 

A further exporatory data analysis (part 2) is done on the training and validation data sets after splitting in order to compare their statistical contents against one another using this noteboook [Exploratory_Data_Analysis_Part2.ipynb](./Exploratory_Data_Analysis_Part2.ipynb). Ideally, each of the data sub-sets should contain all relevant data aspects and show similar distributions of image and object properties as the other data sub-sets. The same would apply in principle for the test set, too, but as the tfrecord files in the test set are pre-defined, we take them as given examples.

### Cross-Validation Concept and Data Split
The training and evaulation pipeline takes two sets of tfrecord files as input data for training and for evaluation, or cross-validation, respectively. If we don't break up all tfrecord files and mix all the images beforehand in order to create new mixed tfrecords for training and evaluation with a better balanced  distribution of data and object features we can only split among the existing tfrecord files. Here we choose the latter and easier option. We will just randomly split into sub-sets among the existing tfrecord files knowing that this may not lead to an optimal balance of data feature distribution in training, validation and test set. As we start with a fully pre-trained object detector just changing the number and types of object classes it should be sufficient to have a smaller training data set. We can leave a bit more for cross-validation. In this case, let's choose a split of approximately 5 : 1. So we randomly split the 97 tfrecord files into 82 files for training and 15 files for evaluation, or cross-validation, respectively. As this project only focuses on exploring some basic capabilities of TensorFlow Object Detection API without having the intention to optimize an object detection model this simple strategy should do for this purpose. However, this simple data strategy is certainly not sufficient for a real application!

The randomly split the data set you can use the `create_splits.py` script by running the following command:
```
python create_splits.py \
    --source_dir {source directory where the set of processed tfrecord files to be split is stored} \
    --target_dir {target directory where a train, val and evtl. test directory is created to store the split data sub-sets}  
    --num_train_files {number of tfrecord files in the training data set} \
    --num_val_files {number of tfrecord files in the validation data set} \
    --num_test_files {number of tfrecord files in the test data set, if not set only a training and validation set will be created}
```
For instance, the following command randomly splits the data set stored in `training_and_validation` into `train` and `val` set with our above defined split ratio:
```
python create_splits.py --source_dir /app/project/data/waymo/training_and_validation --target_dir /app/project/data/waymo --num_train_files 82 --num_val_files 15 --use_symlinks True
```
The `split` function in the `create_splits.py` script does the following:
* create two subfolders: `/app/project/data/waymo/train/` and `/app/project/data/waymo/val/`,
* randomly split the tfrecords files in the source directory between these two folders by symbolically linking the files from `/app/project/data/waymo/training_and_validation` to `/app/project/data/waymo/train/` and `/app/project/data/waymo/val/`.

Important hint:
Please use the absolute paths with respect to your docker root directory. Otherwise the symbolic links might be broken. You can check if there any broken links have created in a subdirectory using the following command:
```
find . -xtype l
```
There will be no output if there are no broken links. Alternatively you can use `ls -l`. It will show you broken links in red color in linux.

After splitting we analyse the data sub-sets again in order to compare the distribution of the training an the valiation data set with one another (s. Dataset Analysis) The training and cross-valiation data sets should be independent but represent the same data domain. Thus, they should have a similar data distribution in order to provide a similar data coverage. This is not exactly the case here. Be we want to keep it simple. Otherwise we need to break up the tfrecord files beforehand, mix all images up contentwise and create new tfrecord files, which requires a larger effort. 

In case all tfrecord files are processed with the same downsampling rate, you can also use `create_splits.py` to randomly split the tfrecord files into `train`, `val` and `test` sub-sets by running the following extended command:

```
python create_splits.py --source_dir /app/project/data/waymo/processed --target_dir /app/project/data/waymo --num_train_files 82 --num_val_files 15 --num_test_files 3
```
The `split` function in the `create_splits.py` file then does the following:
* create three subfolders: `/app/project/data/waymo/train/`, `/app/project/data/waymo/val/`, and `/app/project/data/waymo/test/`
* split the tf records files between these three folders by symbolically linking the files from `/app/project/data/waymo/<source_dir>` to `/app/project/data/waymo/train/`, `/app/project/data/waymo/val/`, and `/app/project/data/waymo/test/`

Please note: If you run `create_splits.py` within the docker environment to create links in the mounted shared memory you need to specify the paths w.r.t. to the docker root directory. However, the links will only work inside the docker container. If you don't like this solution you can also move the files by setting `--use_symlinks False`.


## Model Training and Evaluation Experiments

### Configuration of the Training and Evaluation Pipeline

The TF Object Detection API uses **config files** to set up and parameterize a training, evaluation or inference pipelines. The default configuration we will start with in this project is `pipeline.config`, which is the config for a Single Shot Detector (SSD) based on a Resnet 50 640x640 (RetinaNet50) DNN model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model and weights](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) from TensorFlow Model Zoo and extract the files to `/app/project/experiments/pretrained_models/`.

You need to edit the config file to change the location of the training and validation tfrecord files, as well as the location of the label_map file and the pretrained weights. You also need to adjust the batch size. To do so, e.g. run the following command (adjust the parameters as you need):
```
python edit_config.py --train_dir /app/project/data/waymo/train/ --eval_dir /app/project/data/waymo/val/ --batch_size 4 --checkpoint /app/project/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /app/project/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`. Plase rename it and move it to the corresponding experiments folder (s. Model Training).

### Model Training and Evaluation

We will now do an experiment using Tensorflow Object Detection API. We move the newly created `pipeline_new.config` to a subfolder of our current experiment, e.g. `/app/project/experiments/experiment_0_00` with an index of our choice and rename the pipeline config accoringly, e.g. `pipeline_experiment_0_00.config`. Then we change to the `experiments`folder and launch the training process:
```
python model_main_tf2.py \
  --pipeline_config_path {path to the pipeline.config file} \
  --model_dir {path where the trained model checkpoints are stored} 
```
e.g.
```
python model_main_tf2.py --pipeline_config_path=/app/project/experiments/experiment_0_00/pipeline_experiment_0_00.config --model_dir=/app/project/experiments/experiment_0_00/model_checkpoints
```
Once the training is finished, you can launch an evaluation process in the same terminal in order to obtain an evaluation of the final model checkpoint on the validation data set:
```
python model_main_tf2.py \
  --pipeline_config_path {path to the pipeline.config file} \
  --model_dir {path where the trained model checkpoints are stored} \
  --checkpoint_dir {path where the final trained model checkpoint to be evaluated is stored}  
```
e.g.
```
python model_main_tf2.py --pipeline_config_path=/app/project/experiments/experiment_0_00/pipeline_experiment_0_00.config --model_dir=/app/project/experiments/experiment_0_00/model_checkpoints --checkpoint_dir=/app/project/experiments/experiment_0_00/model_checkpoints
```
If you open up a second command shell you can also launch the evaluation process in parallel to the training process. So says the documentation, at least. However, it seems that this might only be supported within google cloud space. It did not work in my local setup. Evaluation was only executed once the training process was finished. If you run out of GPU resources it might help to switch off visibile GPU devices and use CPU for evaluation. Therefore, you need to set
```
set CUDA_VISIBLE_DEVICES=-1 # use CPU => set no CUDA visible devices
```
or (if above command does not work)
```
set CUDA_VISIBLE_DEVICES="" # use CPU => set no CUDA visible devices
```
before running the evaluation command.

If you don't want to write lengthy commands you can also use the following bash script to ease your life and launch a model training and evaluation process: `train_and_evaluate_model.sh`.

**Note**: Both training and evaluation will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using `CTRL+C` as it will keep on waiting for further checkpoints.

The evaluation metrics can be configured in the `pipeline.config` file in the section `eval_config {}`. The [eval.proto](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/eval.proto) file contains the configurable evaluation options. However, not all of them seem to be supported in all possible cases. By default, the `pipeline.config` that comes with the SSD Resnet 50 640x640 (RetinaNet50) TensorFlow model uses "coco_detection_metrics".
```
eval_config {
  metrics_set: "coco_detection_metrics"
  ...
}
```
Alternatively, "pascal_voc_detection_metrics" are supported, too, and allow for class-wise evaluation.
```
eval_config {
  metrics_set: "pascal_voc_detection_metrics"
  ...
}
```
To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir <path to model checkpoint dir>` or simply `tensorboard --logdir=<path to model checkpoint dir> in the `experiments` folder. It will output a link which you can open in a browser window to visualize the tensor board.


### Training Experiments using Different Options of TF Object Detection API

The experimental model training results are organized as follow:
```
/app/project/experiments/
    - pretrained_models/ - contains the extracted pretrained model files
    - experiment_0_00...09/ - contains trained model checkpoints and evaluation results using SSD Resnet 50 640x640 (RetinaNet50)
    - experiment_1_00...02/ - contains trained model checkpoints and evaluation results using EfficientDet D1 640x640
    - exporter_main_v2.py - script that comes with TF Object Detection API to export a model as a frozen graph
    - export_model.sh - bash script to export a model
    - label_map.pbtxt - label map file covering the classes "vehicle", "pedestrian" and "cyclist"
    - model_main_tf2.py - script that comes with tF Object Detection APIA to launch a training or an evaluation process
    - train_and_evaluate_model.sh - bash script to sequentially run a training and an evaluation process in the same terminal
```
The experimental training and evaluation results are documented in [Experimental_Results.md](./Experimental_Results.md).

### Improve the Performance

When retraining SSD Resnet 50 640x640 (RetinaNet50) on the selected Waymo training set using the three classes "vehicle", "pedestrian" and "cyclist" and the configuration that comes with pretrained model without changing any further options we obtain a clear improvement compared to the initial model on the validation and the test set, but the mean average precision (mAP) do not seem to be optimal. There are still a lot of False Positive and False Negative detections when creating inference videos on arbitrary tfrecord files. Besides using a larger and much better structured or balanced training data set, there are multiple options to improve the training results, e.g.:
* find the optimal optimizer, learning rate and scheduler
* increase number of training steps or epochs
* increase batch size
* increase the size and optimize variety and distribution of the data set
* use augmentation and an adequate data augmentation strategy to increase variety of data contents
* change the model architecture
The config file allows to change hyperparameters like the type of optimizer, learning rate, scheduler and to configure a set of available default augmentation methods. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation methods available in the TensorFlow Object Detection API. In the notebook [Explore_TF_Default_Augmentations.ipynb](./Explore_TF_Default_Augmentations.ipynb) the effect of single default augmentation methods on the raw images are explored and visualized isolated from the training pipeline. In a second notebook [Explore_Augmentations.ipynb](./Explore_Augmentations.ipynb) the effect of different augmentation combinations of the training pipeline are explored and visualized in order to find a constellation that promises and imporovement of the trained model.

The Tensorflow Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many alternative model architectures. The `pipeline.config` file that comes with the pretrained models is unique for each architecture. When chaning the model architecture we need to edit the corresponding `pipeline.config` file and adjust it to our problem (e.g. adapting the number of classes, the training or validation set, etc.).

The experimental results of improving the model performance are documented in [Experimental_Results.md](./Experimental_Results.md).

### Model Inference

#### Export the Trained Model
In order to perform inference without the dependencies of the training environment we need to export a frozen graph of our trained model. THis can be done by running the script `exporter_main_v2.py` in the `experiments`folder. You need to modify the arguments of the following command to adjust it to your models:

```
python exporter_main_v2.py \ 
  --input_type image_tensor \
  --pipeline_config_path {path to the pipeline.config file} \
  --trained_checkpoint_dir {path to the trained model checkpoint} \
  --output_directory {path where to save the exported model} 
```
e.g.
```
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path /app/project/experiments/experiment_0_00/pipeline_experiment_0_00.config --trained_checkpoint_dir /app/project/experiments/experiment_0_00/model_checkpoints --output_directory /app/project/experiments/experiment_0_00/exported_model/
```

This should create a new folder `/app/project/experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model). If you don't want to write lengthy commands you can also use the following bash script to export a model checkpoint as a frozen graph: `export_model.sh`.

####  Creating an Animation

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py \
  --labelmap_path {path to the label map file label_map.pbtxt} \
  --model_path {path to the saved model} \
  --tf_record_path {path to the tfrecord file to infer on} \
  --config_path {path to the pipeline.config file} \
  --output_path {path to store the video e.g. as .mp4 or .gif} 
```
e.g.
```
python inference_video.py --labelmap_path ./label_map.pbtxt --model_path /app/project/experiments/experiment_0_00/exported_model/saved_model/ --tf_record_path /app/project/data/waymo/val/segment-10241508783381919015_2889_360_2909_360_with_camera_labels.tfrecord --config_path /app/project/experiments/experiment_0_00/pipeline_experiment_0_00.config --output_path /app/project/experiments/experiment_0_00/inference/videos/val/segment-10241508783381919015_2889_360_2909_360.mp4
```
If you don't want to write lengthy commands or if you want to create inference videos on a larger set of tfrecord files in a folder you can also use the following bash script: `create_inference_videos.sh`.

## License

### Main Part
Remarks: Since the the code in this repository was build on the [starter code](https://github.com/udacity/nd013-c1-vision-starter) provided by Udacity, it automatically falls under the Udacity license, too:

[LICENSE.md](./LICENSE.md)

This also applies to the changes, extensions or modifications implemented by the code owner, s. [CODEOWNERS](./CODEOWNERS).

### Third Party Part
The tutorial how to explore TFRecordFiles [Explore_Raw_TFRecordFile.ipynb](./third_party/Explore_Raw_TFRecordFile.ipynb) was created based on the tutorials taken over from Waymo open dataset tutorials, and thus, falls under the respective WAYMO license:

[LICENSE_WAYMO-OPEN-DATASET-TUTORIALS](./third_party/LICENSE_WAYMO-OPEN-DATASET-TUTORIALS)

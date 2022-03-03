# Experimental Results

## Data Analysis
The results of the exploratory data analysis on the raw tfrecord data set and the processed tfrecord data set can be visualized using the following jupyter notebooks:
* [Exploratory_Data_Analysis_of_Original_Waymo_Data.ipynb](./Exploratory_Data_Analysis_of_Original_Waymo_Data.ipynb)
* [Exploratory_Data_Analysis_Part1.ipynb](./Exploratory_Data_Analysis_Part1.ipynb)

The first 100 tfrecord files in [filenames.txt](./filenames.txt) used as dataset in this project were recorded at different locations in US in the area in and around San Francisco. The data set contains urban and motorway scenarios with high to sparse traffic density. Most of the scenes have been recorded at daytime under sunny weather conditions, but there are also a few sequences with rather foggy or rainy weather. There are also some videos recorded at dusk/dawn or at nighttime. Maybe the day and nighttime aspect is one of the largest data related contrastive aspects in this data set. Therefore, both training, validation and test data set should contain both scenes at daytime and at nighttime. As you will see when running the exploratory data analysis notebook, this key aspect is given in this case.

## Training, Validation and Test Data
A separate exploratory data analysis has been done on the training, validation and test data sets after splitting in order to compare the statistical distribution of data features or contents. The results are shown in the following notebooks:
* [Exploratory_Data_Analysis_Part2.ipynb](./Exploratory_Data_Analysis_Part2.ipynb) 

Ideally, we should have approximately a similar distribution of relevant data features in all data sub-sets in order to address them both in training, validation and testing. For example, we want to have all data sub-sets contain all object classes - ideally covering similar ranges of different image positions and distances, bounding box sizes and aspect ratio, respectively. We also want to have major environmental changes like daytime/nighttime similarly represented in training, validation and test data set. This is roughly achieved in this case although only a simple random split was used to separate the tfrecord files. Although this is just a sub-optimal solution, the chosen data split should do for the following experiments on exploring the capabilities of TensorFlow Object Detection API. The goal is not to obtain an optimum model, but only to test which pipelin configuration setting achieves what kind of effect on the training results. For this purpose, the data split should be ok.

For the following experiments, the overall data set of 100 tfrecord files was split into 82 training files, 15 validation files and 3 test files. The test files have been pre-selected beforehand (by Udacity), whereas a random split was used to separate the other 97 tfrecord files into the training and validation files. As each tfrecord files contains video sequences recorded with the front camera at a frame rate of 10 Hz, sub-sequent images resemble one another. As the differences between sub-sequent images are rather small, we assume that the contribution to the training is not that much. For the sake of enhancing processing time, the training and validation videos are downsampled by a factor of 10. As each raw tfrecord contains about 198 or 199 images, a downsampled tfrecord file does not contain more than 20 images. Therefore, the data sets for training and validation are far too small to expect very good training results. The training set contains less than 1640 images and the validation set less than 300 images. Using augmentation, we will try to increase the iamge variety and data coverage artificially to some (limited) extend. The outcome of different experiments is documented in the following.

## Data Augmentation
TensorFlow offers a set of in-built augmentation methods. The effect of a selected set of TensorFlow's data augmentation methods is shown in [Explore_TF_Default_Augmentations.ipynb](./Explore_TF_Default_Augmentations.ipynb). In this jupyter notebook, the augmentation methods are directly applied to some sample images without prior image noralization. However, in the preprocessing pipline there are also other in-built steps, which are automatically applied. Some of them are model-dependend like image normalization. The effect of different preprocessor pipeline configurations on a set of sample images can be visualized by running [Explore_Augmentations.ipynb](./Explore_Augmentations.ipynb). The same augmentation settings are later on used in the training experiments.

## Transfer Learning Experiments

### Evaluation of Transfer Learning Experiments
In the following experiments PASCAL VOC metrics are used to evaluate the performance of the re-trained model on the validation set. They allow for a class-wise evalution of the average precision AP@.5IOU metric (meaning AP with IOU threshold = 0.5) obtained from the precision-recall curve as well as the mean average precision value mAP@.5IOU, which is the average precision AP@.5IOU averaged over all classes. Howe these object detection metrics are calculated is shown here: [Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics).

### Exeriments on Optimizer and Batch Size Settings
The first five experiments focus on the optimizer and scheduler settings mainly focusing on the learning rate as well as on testing different batch sizes using the SSD ResNet50 V1 FPN 640x640 (RetinaNet50). In general, larger batch sizes yield better training performance, but are they are limited by the available GPU resources. I am using a local setup with an NVIDIA GeForce GTX 1080 Ti GPU with 12 GB memory.

#### Experiment_0_00

Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
Pipeline configuration: [pipeline_experiment_0_00.config](./experiments/experiment_0_00/pipeline_experiment_0_00.config)
Batch size: 4
Optimizer: momentum optimizer
Scheduler: cosine decay learning rate (learning_rate_base: 0.04, warmup_learning_rate: 0.013333)
Training steps: 25000
Evalution metrics set: "pascal_voc_detection_metrics"
Data augmentation setting: default settings that come with ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
```
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
```

<img src="./experiments/experiment_0_00/results/experiment_0_00_TensorBoard_Screenshot.png" width="700" height="400" />

*Fig. 0.1: Visualization of training and evaluation results on TensorBoards for experiment_0_00*

<img src="./experiments/experiment_0_00/results/experiment_0_00_eval_side_by_side_8_0.png" width="700" height="400" />

*Fig. 0.2: Side-by-side comparison of model inference on a sample image before and after retraining for experiment_0_00*

| Loss function | Loss value | | Evaluation metric | Metric value |
| :----: | :----: | :----: | :----: | :----: |
| Loss/localization_loss | 0.4576 |  | PascalBoxes_Precision/mAP@.5IOU | 0.0394 |
| Loss/classification_loss | 0.4010 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/vehicle | 0.0476 |
| Loss/regularization_loss | 1.0733 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/pedestrian | 0.0312 |
| Loss/total_loss | 1.9319 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/cyclist | nan |

*Table 0.1: Evaluation results on validation set at step 24000 using Pascal VOC metrics for experiment_0_00*

*Discussion:*
* The training data set has 82 tfrecord files with about 20 images each. Therefore, 25000 training steps correspond to approximately 15 epochs (passes through over the whole data set not considering the artificial enrichment of the training data set by augmentation). The number of training steps will be kept constant in the following for the sake of comparability.
* Starting from an initial learning rate the cosine decay scheduler ramps up to a rather large burn-in learning rate. Afterwards, the learning rate drops gradually down to zero allowing the optimizer to better converge to a local minimum.
* The evaluation loss on the final checkpoint seems to be not too far off from the training loss. Therefore, I assume the model is not overfitting to the training data set yet.
* Although the model shows a better performance on the validation data after training than the initial pre-trained model - especially in detecting pedestrians (s. fig. 0.2), the overall mAP performance seems to be quite low. So there is still lots of room for improvements.
* Although there are a few cyclists in the validation data set, they are not detected with sufficient IOU threshold and confidence. As cyclists are also underrepresented in the training data set they are proabably not trained well enough. An increase of images with representative cyclist objects would help to improve.

#### Experiment_0_01

Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
Pipeline configuration: [pipeline_experiment_0_01.config](./experiments/experiment_0_01/pipeline_experiment_0_01.config)
Batch size: 4
Optimizer: momentum optimizer
Scheduler: constant learning rate
Learning_rate: 0.01
Training steps: 25000
Evalution metrics set: "pascal_voc_detection_metrics"
Data augmentation setting: default settings that come with ssd_resnet50_v1_fpn_640x640_coco17_tpu-8

<img src="./experiments/experiment_0_01/results/experiment_0_01_TensorBoard_Screenshot.png" width="700" height="400" />

*Fig. 1.1: Visualization of training and evaluation results on TensorBoards for experiment_0_01*

<img src="./experiments/experiment_0_01/results/experiment_0_01_eval_side_by_side_8_0.png" width="700" height="400" />

*Fig. 1.2: Side-by-side comparison of model inference on a sample image before and after retraining for experiment_0_01*

| Loss function | Loss value | | Evaluation metric | Metric value |
| :----: | :----: | :----: | :----: | :----: |
| Loss/localization_loss | 0.4154 |  | PascalBoxes_Precision/mAP@.5IOU | 0.0646 |
| Loss/classification_loss | 0.3568 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/vehicle | 0.0585 |
| Loss/regularization_loss | 0.1836 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/pedestrian | 0.0707 |
| Loss/total_loss | 0.9558 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/cyclist | nan |

*Table 1.1: Evaluation results on validation set at step 24000 using Pascal VOC metrics for experiment_0_01*

*Discussion:*
* Using a constant learning rate of 0.01 instead of a cosine decay learning rate leads to a lower loss after 25000 steps. Therefore, the base learning rate of the pervious setting was too high for this optimization problem.
* The evaluation loss on the final checkpoint is very close to the training loss. So there is no overfitting.
* The model shows better performance after training than the initial pre-trained model - especially in detecting pedestrians (s. fig. 1.2), but the overall mAP@.5IOU performance seems to be still very low.

#### Experiment_0_02

Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
Pipeline configuration: [pipeline_experiment_0_02.config](./experiments/experiment_0_02/pipeline_experiment_0_02.config)
Batch size: 8
Optimizer: momentum optimizer
Scheduler: constant learning rate
Learning_rate: 0.01
Training steps: 25000
Evalution metrics set: "pascal_voc_detection_metrics"
Data augmentation setting: default settings that come with ssd_resnet50_v1_fpn_640x640_coco17_tpu-8

<img src="./experiments/experiment_0_02/results/experiment_0_02_TensorBoard_Screenshot.png" width="700" height="400" />

*Fig. 2.1: Visualization of training and evaluation results on TensorBoards for experiment_0_02*

<img src="./experiments/experiment_0_02/results/experiment_0_02_eval_side_by_side_8_0.png" width="700" height="400" />

*Fig. 2.2: Side-by-side comparison of model inference on a sample image before and after retraining for experiment_0_02*

| Loss function | Loss value | | Evaluation metric | Metric value |
| :----: | :----: | :----: | :----: | :----: |
| Loss/localization_loss | 0.4158 |  | PascalBoxes_Precision/mAP@.5IOU | 0.1153 |
| Loss/classification_loss | 0.3721 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/vehicle | 0.0723 |
| Loss/regularization_loss | 0.1341 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/pedestrian | 0.1584 |
| Loss/total_loss | 0.9220 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/cyclist | nan |

*Table 2.1: Evaluation results on validation set at step 24000 using Pascal VOC metrics for experiment_0_02*

*Discussion:*
* When looking at the training loss curve over the number of training steps a larger peak is striking out. Here, loss skyrockets quickly because the momentum optimizer might have hopped out of a local minimum due to too high a learning rate or too high a momentum value. Lowering the learning rate or a different optimizer might yield better results.
* A larger batch size reduces the training and evaluation loss and yields a noticable improvement of the mAP@.5IOU performance.
* Evaluation and training loss are almost identical after 24000 steps, which is quit good.
* However, in general model performance w.r.t. mAP@.5IOU is still quite low.

#### Experiment_0_03

Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
Pipeline configuration: [pipeline_experiment_0_03.config](./experiments/experiment_0_03/pipeline_experiment_0_03.config)
Batch size: 4
Optimizer: adam optimizer
Scheduler: constant learning rate
Learning_rate: 0.01
Training steps: 25000
Evalution metrics set: "pascal_voc_detection_metrics"
Data augmentation setting: default settings that come with ssd_resnet50_v1_fpn_640x640_coco17_tpu-8

<img src="./experiments/experiment_0_03/results/experiment_0_03_TensorBoard_Screenshot.png" width="700" height="400" />

*Fig. 3.1: Visualization of training and evaluation results on TensorBoards for experiment_0_03*

<img src="./experiments/experiment_0_03/results/experiment_0_03_eval_side_by_side_8_0.png" width="700" height="400" />

*Fig. 3.2: Side-by-side comparison of model inference on a sample image before and after retraining for experiment_0_03*

| Loss function | Loss value | | Evaluation metric | Metric value |
| :----: | :----: | :----: | :----: | :----: |
| Loss/localization_loss | 0.9672 |  | PascalBoxes_Precision/mAP@.5IOU | 0.0004 |
| Loss/classification_loss | 13.1453 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/vehicle | 0.0008 |
| Loss/regularization_loss | 0.2624 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/pedestrian | 0.0000 |
| Loss/total_loss | 14.3748 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/cyclist | nan |

*Table 3.1: Evaluation results on validation set at step 24000 using Pascal VOC metrics for experiment_0_03*

*Discussion:* 
* When using Adam optimizer instead of momentum optimizer with a batch size of 4 in both cases (compare against experiment 0.01), training loss still slowly decreases converging to a kind of steady state, but we do not get an improvement. In contrast, the loss (especially classification loss) increases and the performance drops dramatically. Thus, Adam optimizer seems to be less suited for this optimization problem.
* So we will stick with momentum optimizer. It seems to be the better choice for this optimization problem.

#### Experiment_0_04

Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
Pipeline configuration: [pipeline_experiment_0_04.config](./experiments/experiment_0_04/pipeline_experiment_0_04.config)
Batch size: 8
Optimizer: momentum optimizer
Scheduler: manual step learning rate
```
  optimizer {
    momentum_optimizer {
      learning_rate {
        manual_step_learning_rate {
          initial_learning_rate: 0.01
          schedule {
            step: 8000
            learning_rate: 0.005
          }
          schedule {
            step: 12000
            learning_rate: 0.0025
          }
          schedule {
            step: 16000
            learning_rate: 0.00125
          }
          schedule {
            step: 20000
            learning_rate: 0.000625
          }
        }
      }
      # momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
```
Training steps: 25000
Evalution metrics set: "pascal_voc_detection_metrics"
Data augmentation setting: default settings that come with ssd_resnet50_v1_fpn_640x640_coco17_tpu-8

<img src="./experiments/experiment_0_04/results/experiment_0_04_TensorBoard_Screenshot.png" width="700" height="400" />

*Fig. 4.1: Visualization of training and evaluation results on TensorBoards for experiment_0_04*

<img src="./experiments/experiment_0_04/results/experiment_0_04_eval_side_by_side_8_0.png" width="700" height="400" />

*Fig. 4.2: Side-by-side comparison of model inference on a sample image before and after retraining for experiment_0_04*

| Loss function | Loss value | | Evaluation metric | Metric value |
| :----: | :----: | :----: | :----: | :----: |
| Loss/localization_loss | 0.3929 |  | PascalBoxes_Precision/mAP@.5IOU | 0.1113 |
| Loss/classification_loss | 0.3590 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/vehicle | 0.0677 |
| Loss/regularization_loss | 0.2263 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/pedestrian | 0.1548 |
| Loss/total_loss | 0.9783 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/cyclist | nan |

*Table 4.1: Evaluation results on validation set at step 24000 using Pascal VOC metrics for experiment_0_04*

*Discussion:* 
* A manual step learning rate is applied starting at an initial learnign rate of 0.01. It allows a gradual reduction of the learning rate in user-defined intervalls w.r.t. the number of training steps. If properly set according to the optimization problem at hand, it can lead to a better convergence to local minimum but may also get trapped in a sub-optimal local minimum too early
* Compared to experiment_0_02 with the same setting except for the constant learning rate, there neither larger improvements nor larger performance drops. The performance on vehicles is slightly improved, but the performance on pedestrians is slightly decreased compared to experiment_0_02.

#### Experiment_0_05

Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
Pipeline configuration: [pipeline_experiment_0_05.config](./experiments/experiment_0_05/pipeline_experiment_0_05.config)
Batch size: 8
Optimizer: momentum optimizer
Scheduler: exponential decay learning rate
```
  optimizer {
    momentum_optimizer {
      learning_rate {
        exponential_decay_learning_rate {
          initial_learning_rate: 0.01
          decay_steps: 500
          decay_factor: 0.9
          staircase: true
          min_learning_rate: 0.000051538
          burnin_learning_rate: 0.0
          burnin_steps: 0
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
```
Training steps: 25000
Evalution metrics set: "pascal_voc_detection_metrics"
Data augmentation setting: default settings that come with ssd_resnet50_v1_fpn_640x640_coco17_tpu-8

<img src="./experiments/experiment_0_05/results/experiment_0_05_TensorBoard_Screenshot.png" width="700" height="400" />

*Fig. 5.1: Visualization of training and evaluation results on TensorBoards for experiment_0_05*

<img src="./experiments/experiment_0_05/results/experiment_0_05_eval_side_by_side_8_0.png" width="700" height="400" />

*Fig. 5.2: Side-by-side comparison of model inference on a sample image before and after retraining for experiment_0_05*

| Loss function | Loss value | | Evaluation metric | Metric value |
| :----: | :----: | :----: | :----: | :----: |
| Loss/localization_loss |  0.4644 |  | PascalBoxes_Precision/mAP@.5IOU | 0.0544 |
| Loss/classification_loss | 0.3793 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/vehicle | 0.0544 |
| Loss/regularization_loss | 0.5891 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/pedestrian | 0.0545 |
| Loss/total_loss | 1.4328 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/cyclist | nan |

*Table 5.1: Evaluation results on validation set at step 24000 using Pascal VOC metrics for experiment_0_05*

*Discussion:* 
* In principle, the same settings are used in experiment_0_05 compared to experiment_0_04, except for the learning rate decay function. In this experiment, an exponential decay is applied for the learning rate starting at an initial learning rate of 0.01.
* Compared to experiment_0_04, the results are not improved. It could be that the learning rate was decreased too soon and too low such that the optimization got caught in a sub-optimal local minimum too early. Hence, the learning rate setting from experiment_0_04 was slightly better.

### Apply Data Augmentation
The different standard augmentation methods of TensorFlow target on increasing the variety of different aspects, which can be distinguished as follows, for instance:
* augmentation methods like random cropping, resizing, flipping, or rotation that try to increase variety of object sizes and object locations on image plane
* augmentation methods that randomly adjust hue, saturation, brightness, or contrast or to randomly distort image colors within some boundaries in order to increase the variety of lighting effects
* augmentation methods like putting a random jitter on the bounding boxes to simulated labeling errors, for example
Above augmentation methods are tested in the following experiments in order to artificially increase the variety of data samples in our rather very small training and validation data set.

All augmentation methods are applied randomly with a set probability. If multiple augmentation methods are configured the probability is adapted according to the number of applied augmentation methods if not explicitely set. Thus, the more we augment the less original images the model will see during training. Normally, the number of steps should be increased if data augmentation is used on top. However, we will use the same number of training steps in all experiments.

#### Experiment_0_06

Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
Pipeline configuration: [pipeline_experiment_0_06.config](./experiments/experiment_0_06/pipeline_experiment_0_06.config)
Batch size: 8
Optimizer: momentum optimizer
Scheduler: manual step learning rate as in experiment_0_04
Training steps: 25000
Evalution metrics set: "pascal_voc_detection_metrics"
Data augmentation setting:
```
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  data_augmentation_options {
    random_rgb_to_gray {
    }
  }
  data_augmentation_options {
    random_adjust_brightness {
      max_delta: 0.3
    }
  }
  data_augmentation_options {
    random_adjust_saturation {
      min_delta: 0.7
      max_delta: 1.3
    }
  }
  data_augmentation_options {
    random_adjust_hue {
      max_delta: 0.05
    }
  }
  data_augmentation_options {
    random_adjust_contrast {
      min_delta: 0.5
      max_delta: 1.5
    }
  }
  data_augmentation_options {
    random_jitter_boxes {
      ratio: 0.05
      jitter_mode: 0
    }
  }
```

<img src="./experiments/experiment_0_06/results/experiment_0_06_TensorBoard_Screenshot.png" width="700" height="400" />

*Fig. 6.1: Visualization of training and evaluation results on TensorBoards for experiment_0_06*

<img src="./experiments/experiment_0_06/results/experiment_0_06_eval_side_by_side_8_0.png" width="700" height="400" />

*Fig. 6.2: Side-by-side comparison of model inference on a sample image before and after retraining for experiment_0_06*

| Loss function | Loss value | | Evaluation metric | Metric value |
| :----: | :----: | :----: | :----: | :----: |
| Loss/localization_loss | 0.4522 |  | PascalBoxes_Precision/mAP@.5IOU | 0.0568 |
| Loss/classification_loss | 0.3577 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/vehicle | 0.0498 |
| Loss/regularization_loss | 0.5723 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/pedestrian | 0.0637 |
| Loss/total_loss | 1.3822 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/cyclist | nan |

*Table 6.1: Evaluation results on validation set at step 24000 using Pascal VOC metrics for experiment_0_06*

*Discussion:* 
* Besides random image crop and random horizontal flip further image augmentation methdos are applied focusing on enlarging the spectrum of hue, saturation, brightness, or contrast, for example, plus adding some jitter on the bounding boxes to simulate labeling errors.
* Actually, it was expected that color-based augmentation might noticably improve the training results. However, the overall results only improve a tiny bit compared to experiment_0_04. The AP@.5IOU performance on vehicles even decreases a little, whereas the AP@.5IOU performance on pedestrians slightly increases.
* A possible reason can be that the measure of adding random jitter on the bounding boxes acts contraproductive and makes things rather worse as it deteriorates ground truth. A slightly higher localization loss compared to experiment_0_04 can be interpreted as another supporting indicator for this hypothesis.
* It is also possible that we need to increase the number of training steps when applying augmentation.
* AP@.5IOU performance on cyclists is still NaN leading to the conclusion that they are underrepresented in the training data such they are not adequately learned by the model.

#### Experiment 0.07

Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
Pipeline configuration: [pipeline_experiment_0_07.config](./experiments/experiment_0_07/pipeline_experiment_0_07.config)
Batch size: 8
Optimizer: momentum optimizer
Scheduler: manual step learning rate as in experiment_0_04 and experiment_0_06
Training steps: 25000
Evalution metrics set: "pascal_voc_detection_metrics"
Data augmentation setting:
```
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  data_augmentation_options {
    random_rgb_to_gray {
    }
  }
  data_augmentation_options {
    random_distort_color {
      color_ordering: 0
    }
  }
  data_augmentation_options {
    random_jitter_boxes {
      ratio: 0.05
    }
  }
```

<img src="./experiments/experiment_0_07/results/experiment_0_07_TensorBoard_Screenshot.png" width="700" height="400" />

*Fig. 7.1: Visualization of training and evaluation results on TensorBoards for experiment_0_07*

<img src="./experiments/experiment_0_07/results/experiment_0_07_eval_side_by_side_8_0.png" width="700" height="400" />

*Fig. 7.2: Side-by-side comparison of model inference on a sample image before and after retraining for experiment_0_07*

| Loss function | Loss value | | Evaluation metric | Metric value |
| :----: | :----: | :----: | :----: | :----: |
| Loss/localization_loss | 0.4336 |  | PascalBoxes_Precision/mAP@.5IOU | 0.0707 |
| Loss/classification_loss | 0.3381 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/vehicle | 0.0545 |
| Loss/regularization_loss | 0.8293 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/pedestrian | 0.0870 |
| Loss/total_loss | 1.6010 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/cyclist | nan |

*Table 7.1: Evaluation results on validation set at step 24000 using Pascal VOC metrics for experiment_0_07*

*Discussion:* 
* Instead of randomly augmenting hue, saturation, brightness, or contrast individually, we apply the in-built random color distortion measure, which does this by itself automatically using pre-defined ranges.
* Here we have a slight improvement compared to experiment_0_06. The automatic color distortion setting might yield a little advantage here.


#### Experiment 0.08

Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
Pipeline configuration: [pipeline_experiment_0_08.config](./experiments/experiment_0_08/pipeline_experiment_0_08.config)
Batch size: 8
Optimizer: momentum optimizer
Scheduler: manual step learning rate as in experiment_0_04, experiment_0_06 and experiment_0_07
Training steps: 25000
Evalution metrics set: "pascal_voc_detection_metrics"
Data augmentation setting:
```
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_vertical_flip {
    }
  }
  data_augmentation_options {
    random_rotation90 {
    }
  }
  data_augmentation_options {
    random_rotation90 {
    }
  }
  data_augmentation_options {
    random_rotation90 {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
```

<img src="./experiments/experiment_0_08/results/experiment_0_08_TensorBoard_Screenshot.png" width="700" height="400" />

*Fig. 8.1: Visualization of training and evaluation results on TensorBoards for experiment_0_08*

<img src="./experiments/experiment_0_08/results/experiment_0_08_eval_side_by_side_8_0.png" width="700" height="400" />

*Fig. 8.2: Side-by-side comparison of model inference on a sample image before and after retraining for experiment_0_08*

| Loss function | Loss value | | Evaluation metric | Metric value |
| :----: | :----: | :----: | :----: | :----: |
| Loss/localization_loss | 0.3624 |  | PascalBoxes_Precision/mAP@.5IOU | 0.1540 |
| Loss/classification_loss | 0.2656 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/vehicle | 0.0726 |
| Loss/regularization_loss | 0.1826 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/pedestrian | 0.2354 |
| Loss/total_loss | 0.8106 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/cyclist | nan |

*Table 8.1: Evaluation results on validation set at step 24000 using Pascal VOC metrics for experiment_0_08*

*Discussion:* 
* In experiment_0_08 we don't use color distortion. Instead, we focus on cropping, rotating and flipping the images, which changes the object sizes and aspect ratios as well as their location.
* Here we get a noticable improvement compared to experiment_0_04, experiment_0_06 or experiment_0_07 - especially w.r.t. AP@.5IOU on pedestrians. Changing the object size, aspect ratio or location seems to have a stronger effect on the training as color distortions.


### Evaluation Metrics

#### Experiment 0.09

Pretrained model: ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
Pipeline configuration: [pipeline_experiment_0_09.config](./experiments/experiment_0_09/pipeline_experiment_0_09.config)
Batch size: 8
```
  optimizer {
    momentum_optimizer {
      learning_rate {
        exponential_decay_learning_rate {
          initial_learning_rate: 0.01
          decay_steps: 5000
          decay_factor: 0.5
          staircase: true
          burnin_learning_rate: 0.0
          burnin_steps: 0
          min_learning_rate: 0.0003125
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
```
Training steps: 25000
Evalution metrics set: "pascal_voc_detection_metrics"
Data augmentation setting: same as in experiment_0_06

<img src="./experiments/experiment_0_09/results/experiment_0_09_TensorBoard_Screenshot.png" width="700" height="400" />

*Fig. 9.1: Visualization of training and evaluation results on TensorBoards for experiment_0_09*

<img src="./experiments/experiment_0_09/results/experiment_0_09_eval_side_by_side_8_0.png" width="700" height="400" />

*Fig. 9.2: Side-by-side comparison of model inference on a sample image before and after retraining for experiment_0_09*

| Loss function | Loss value | | Evaluation metric | Metric value |
| :----: | :----: | :----: | :----: | :----: |
| Loss/localization_loss | 0.3496|  | PascalBoxes_Precision/mAP@.5IOU | 0.1492 |
| Loss/classification_loss | 0.2860 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/vehicle | 0.0728 |
| Loss/regularization_loss | 0.1932 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/pedestrian | 0.2257 |
| Loss/total_loss | x.x | 0.8288 | PascalBoxes_PerformanceByCategory/AP@.5IOU/cyclist | nan |

*Table 9.1: Evaluation results on validation set at step 24000 using Pascal VOC metrics for experiment_0_09*

*Discussion:* 
* Again we re-use the augmentation setting from experiment_0_06 focusing on color changes
* However, this time we change the learning rate scheduling using exponential decay rate instead of manual step decay.
* This time we see a noticable improvement. Obviously, the effect of different settings cannot be so easily predicted due to the complexity of the optimization problem. Applying a method that shows hardly an effect in one case might have  larger effect in another case. As it looks like, we have found a better local minimum of our extremely high-dimensional optimization problem.


### Alternative Models

According to the model specifications given on [tf2_detection_zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), deeper machine learning models or models trained on higher resoultion images tend to provide better performance w.r.t. mean average precision (mAP), but they also need more computational resources, and thus, feed forward processing takes longer. Our initial model SSD ResNet50 V1 FPN 640x640 (RetinaNet50) is rather a lighter weighted candidate compared to the others with a moderate mAP value on COCO data set. Therefore, it needs a comparatively low processing time. In order to look for a better alternative, we are searching a model with similar resolution and processing time, but with better mAP values. EfficientDet D1 640x640 seems to be a good candidate. Therefore, some more experiments have been done with this model. 

**Please note:** EfficientDet D1 640x640 uses a different normalization method than SSD ResNet50 V1 FPN 640x640 (RetinaNet50).

#### Experiment 1.00

Pretrained model: efficientdet_d1_coco17_tpu-32
Pipeline configuration: [pipeline_experiment_1_00.config](./experiments/experiment_1_00/pipeline_experiment_1_00.config)
Batch size: 8
Optimizer: momentum optimizer
Scheduler: constant learning rate
Learning rate: 0.005
Training steps: 30000 (slightly increased number of training steps)
Evalution metrics set: "pascal_voc_detection_metrics"
Data augmentation settings: default settings from efficientdt_d1_coco17_tpu-32
```
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_scale_crop_and_pad_to_square {
      output_size: 640
      scale_min: 0.10000000149011612
      scale_max: 2.0
    }
  }
```

<img src="./experiments/experiment_1_00/results/experiment_1_00_TensorBoard_Screenshot.png" width="700" height="400" />

*Fig. 10.1: Visualization of training and evaluation results on TensorBoards for experiment_1_00*

<img src="./experiments/experiment_1_00/results/experiment_1_00_eval_side_by_side_8_0.png" width="700" height="400" />

*Fig. 10.2: Side-by-side comparison of model inference on a sample image before and after retraining for experiment_1_00*

| Loss function | Loss value | | Evaluation metric | Metric value |
| :----: | :----: | :----: | :----: | :----: |
| Loss/localization_loss | 0.0160 |  | PascalBoxes_Precision/mAP@.5IOU | 0.1389 |
| Loss/classification_loss | 0.2839 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/vehicle | 0.0678 |
| Loss/regularization_loss | 0.0286 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/pedestrian | 0.2867 |
| Loss/total_loss | 0.3285 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/cyclist | 0.0623 |

*Table 10.1: Evaluation results on validation set at step 30000 using Pascal VOC metrics for experiment_1_00*

*Discussion:* 
* Some initial experiments with the changed model shwoed that the efficientdet_d1_coco17_tpu-32 model requires a lower step size than ssd_resnet50_v1_fpn_640x640_coco17_tpu-8. In our first documented experiment we use a learning rate of 0.005, which is 50% of the previous one.
* Compared to experiment_0_00, the model change (including the adapted learning rate) leads to a noticably higher mAP@.5IOU performance.
* As the training process has not fully converged the slightly increased number of training steps has a little contribution on the improvement.
* This time we also have some cyclists detected with sufficient confidence and IOU in order to contribute to a metric value other than NaN.


#### Experiment 1.01

Pretrained model: efficientdet_d1_coco17_tpu-32
Pipeline configuration: [pipeline_experiment_1_01.config](./experiments/experiment_1_01/pipeline_experiment_1_01.config)
Batch size: 8
Optimizer: adam optimizer
Scheduler: constant learning rate of 0.005
```
  optimizer {
    adam_optimizer {
      learning_rate {
        constant_learning_rate {
          learning_rate: 0.005
        }
      }
      epsilon: 1e-8
    }
    use_moving_average: false
  }
```
Learning rate: 0.005
Training steps: 25000
Evalution metrics set: "pascal_voc_detection_metrics"
Data augmentation settings: default settings from efficientdt_d1_coco17_tpu-32 as in experiment_1_00

<img src="./experiments/experiment_1_01/results/experiment_1_01_TensorBoard_Screenshot.png" width="700" height="400" />

*Fig. 11.1: Visualization of training and evaluation results on TensorBoards for experiment_1_01*

<img src="./experiments/experiment_1_01/results/experiment_1_01_eval_side_by_side_8_0.png" width="700" height="400" />

*Fig. 11.2: Side-by-side comparison of model inference on a sample image before and after retraining for experiment_1_01*

| Loss function | Loss value | | Evaluation metric | Metric value |
| :----: | :----: | :----: | :----: | :----: |
| Loss/localization_loss | 0.0183 |  | PascalBoxes_Precision/mAP@.5IOU | 0.0725 |
| Loss/classification_loss | 0.3915 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/vehicle | 0.0584 |
| Loss/regularization_loss | 0.0417 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/pedestrian | 0.1582|
| Loss/total_loss | 0.4514 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/cyclist | 0.0008 |

*Table 11.1: Evaluation results on validation set at step 24000 using Pascal VOC metrics for experiment_1_01*

*Discussion:* 
* As in experiment_0_01 we also want to know if an Adam optimizer might yield better training results, but this is not the case. So momentum optimizer is also the better choice for the efficientdet_d1_coco17_tpu-32 model


#### Experiment 1.02

Pretrained model: efficientdet_d1_coco17_tpu-32
Pipeline configuration: [pipeline_experiment_1_02.config](./experiments/experiment_1_02/pipeline_experiment_1_02.config)
Batch size: 8
Optimizer: momentum optimizer
Scheduler: manual step learning rate
```
  optimizer {
    momentum_optimizer {
      learning_rate {
        manual_step_learning_rate {
          initial_learning_rate: 0.005
          schedule {
            step: 8000
            learning_rate: 0.0025
          }
          schedule {
            step: 12000
            learning_rate: 0.00125
          }
          schedule {
            step: 16000
            learning_rate: 0.000625
          }
          schedule {
            step: 20000
            learning_rate: 0.0003125
          }
        }
      }
      # momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
```
Learning rate: 0.005
Evalution metrics set: "pascal_voc_detection_metrics"
Data augmentation settings: similar to experiment_0_09
```
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_scale_crop_and_pad_to_square {
      output_size: 640
      scale_min: 0.10000000149011612
      scale_max: 2.0
    }
  }
  data_augmentation_options {
    random_rgb_to_gray {
    }
  }
  data_augmentation_options {
    random_adjust_brightness {
      max_delta: 0.3
    }
  }
  data_augmentation_options {
    random_adjust_saturation {
      min_delta: 0.7
      max_delta: 1.3
    }
  }
  data_augmentation_options {
    random_adjust_hue {
      max_delta: 0.05
    }
  }
  data_augmentation_options {
    random_adjust_contrast {
      min_delta: 0.5
      max_delta: 1.5
    }
  }
  data_augmentation_options {
    random_jitter_boxes {
      ratio: 0.05
      jitter_mode: 0
    }
  }
```

<img src="./experiments/experiment_1_02/results/experiment_1_02_TensorBoard_Screenshot.png" width="700" height="400" />

*Fig. 12.1: Visualization of training and evaluation results on TensorBoards for experiment_1_02*

<img src="./experiments/experiment_1_02/results/experiment_1_02_eval_side_by_side_8_0.png" width="700" height="400" />

*Fig. 12.2: Side-by-side comparison of model inference on a sample image before and after retraining for experiment_1_02*

| Loss function | Loss value | | Evaluation metric | Metric value |
| :----: | :----: | :----: | :----: | :----: |
| Loss/localization_loss | 0.0168 |  | PascalBoxes_Precision/mAP@.5IOU | 0.1260 |
| Loss/classification_loss | 0.2696 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/vehicle | 0.0697 |
| Loss/regularization_loss | 0.0292 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/pedestrian | 0.2756 |
| Loss/total_loss | 0.3155 |  | PascalBoxes_PerformanceByCategory/AP@.5IOU/cyclist | 0.0328 |

*Table 12.1: Evaluation results on validation set at step 24000 using Pascal VOC metrics for experiment_1_02*

*Discussion:* 
* Here momentum optimizer is used again as in experiment_1_00, but this time with a decaying learning rate using manual steps.
* Additionally, augmentation on the color w.r.t. brightness, hue, saturation, or contrast, and a bounding box jitter is applied, but this does not improve the performance compared to experiment_1_00.

### Summary
In summary, a set of transfer learning experiments has been done using two different SSD models re-trained on a very small selected Waymo training data set. In general, the overall performance on our very small training data set is rather low. If much more data had been used the results would surely be better. This could be achieved by using the whole available Waymo dataset and not just a sub-set. With respect to data augmentation, cropping and resizing, flipping and rotation have a stronger effect in inducing variety than just augmentation of color values as this also increases the variety of the trained spectrum of SSD anchor box aspect ratios.
As we do not target optimizing the performance of a SSD model in this project, but rather experimenting with TensorFlow Object Detection API, model performnce is of second rank. Over all conducted experiments, experiment_0_08 and experiment_1_00 showed the best results on validation set so far. What is more, compared to eperiment_0_00...09 we also have a nonzero AP@.5IOU detection performance for cyclists in experiment_1_00...02.

## Inference

### Export the model and infer on test video sequences
In order to get a visual grasp of the achieved detection performance inference videos are created on the tfrecord files. Therefore, the final model checkpoint of each experiment is exported as a frozen graph in order to perform inference on some test videos without the need of having a full TensorFlow training environment installed.

#### Experiment_1_00
As the re-trained model from experiment_1_00 has shown the best overall results compared to the other experiments including detection of cyclists, some inference videos are provided as examples in this repository. Please follow the links below to watch them.

Video example 1: [Inference on test video sequence id 10072231702153043603 (downsampled by a factor of 10) using exported model from experiment_1_00](./experiments/experiment_1_00/inference/videos/test/segment-10072231702153043603_5725_000_5745_000.mp4)

In the first test video sequence, we have a country road scenario with only a moderate number of other vehicles. Detection works in principle, but some objects are missed - especially when they are rather further away from our ego vehicle. If they come closer detection performance increases. Although most of the objects in the video sequences are detected correctly, detection confidence is quite low, or lower than 50 %, respectively. This should be improved. Adding more suitable data samples would help.

Video example 2: [Inference on test video sequence id 12012663867578114640 (downsampled by a factor of 10) using exported model from experiment_1_00](./experiments/experiment_1_00/segment-12012663867578114640_820_000_840_000.mp4)

The second test video sequence shows a complex urban scenario with many traffic participants. Object detection seems to work well in principle here, but detection performance is quite low for some objects (< 50 %), which means the detected objects are likely not considered objects of the respective class depending on the thresholds. This should to be improved, for course.

Video example 3: [Inference on test video sequence id 12200383401366682847 (downsampled by a factor of 10) using exported model from experiment_1_00](./experiments/experiment_1_00/segment-12200383401366682847_2552_140_2572_140.mp4)

The third test video sequences shwos an urban scenario at night with a moderate number of traffic participants. Most of the objects are detected, except for some oncoming cars with their headlights on. They are often detected quite late compared to parking cars, which are not so safety relevant. It is good to see that object detection also works in principle at nighttime, however, some relevant objects are overlooked and a non-neglectible number of objects are detected with a confidence less than 50 %. So there is quite some room for further improvements. The major leverage would be in this case to significantly increase the training data set and its variety and coverage with respect to target domain.

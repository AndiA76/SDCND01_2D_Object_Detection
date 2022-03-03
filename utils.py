# Copyright (c) 2022 by Andreas Albrecht
# Copyright (c) 2012 - 2021, Udacity, Inc.
# All rights reserved.
# This file is part of the computer vision course project submission of Udacity's Self-driving
# Cars Nanodegree Program. It has been developed based on the starter code provided by Udacity
# on https://github.com/udacity/nd013-c1-vision-starter, and thus, is released under "Udacity
# License Agreement". Please see LICENSE file, which should be included as part of this package.

import logging

import tensorflow.compat.v1 as tf
from object_detection.inputs import train_input
from object_detection.protos import input_reader_pb2
from object_detection.utils import label_map_util
from object_detection.builders.dataset_builder import build as build_dataset
from object_detection.utils.config_util import get_configs_from_pipeline_file
from waymo_open_dataset import dataset_pb2 as open_dataset


def get_label_map(label_map_path='label_map.pbtxt'):
    """
    Get label map as category index and class name index
    args:
      - label_map_path [str]: full label map filepath (defaults to 'label_map.pbtxt')
    returns:
      - category_index [dict]: label map as category index
      - classname_index [dict]: inverse label map as class name index
    """
    # load label map
    label_map = label_map_util.load_labelmap(label_map_path)
    # get label map as classname index from label map
    classname_index = label_map_util.get_label_map_dict(label_map)
    # invert keys and values in dictionary
    category_index = dict()
    for key, value in classname_index.items():
        category_index.update({value : key})
    # retunr category index and classname index
    return category_index, classname_index


def get_dataset(tfrecord_path, label_map='label_map.pbtxt'):
    """
    Opens a tf record file and create tf dataset
    args:
      - tfrecord_path [str]: path to a tf record file
      - label_map [str]: path the label_map file
    returns:
      - dataset [tf.Dataset]: tensorflow dataset
    """
    input_config = input_reader_pb2.InputReader()
    input_config.label_map_path = label_map
    input_config.tf_record_input_reader.input_path[:] = [tfrecord_path]
    
    dataset = build_dataset(input_config)
    return dataset


def get_module_logger(mod_name):
    """ simple logger """
    logger = logging.getLogger(mod_name)
    #handler = logging.StreamHandler()
    handler = logging.FileHandler('debug.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def get_train_input(config_path):
  """
  Get the tf dataset that inputs training batches
  args:
    - config_path [str]: path to the edited config file
  returns:
    - dataset [tf.Dataset]: data outputting augmented batches
  """
  # parse config
  configs = get_configs_from_pipeline_file(config_path)
  train_config = configs['train_config']
  train_input_config = configs['train_input_config']

  # get the dataset
  dataset = train_input(train_config, train_input_config, configs['model'])
  return dataset


def parse_frame(frame, camera_name='FRONT'):
    """ 
    take a frame, output the bboxes and the image

    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
      for data in dataset:
      frame = open_dataset.Frame()
      frame.ParseFromString(bytearray(data.numpy()))
    
    args:
      - frame [waymo_open_dataset.dataset_pb2.Frame]: a waymo frame, contains images and annotations
      - camera_name [str]: one frame contains images and annotations for multiple cameras
    
    returns:
      - encoded_jpeg [bytes]: jpeg encoded image
      - annotations [protobuf object]: bboxes and classes
    """
    # get image
    images = frame.images
    for im in images:
        if open_dataset.CameraName.Name.Name(im.name) != camera_name:
            continue
        encoded_jpeg = im.image
    
    # get bboxes
    labels = frame.camera_labels
    for lab in labels:
        if open_dataset.CameraName.Name.Name(lab.name) != camera_name:
            continue
        annotations = lab.labels
    return encoded_jpeg, annotations


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
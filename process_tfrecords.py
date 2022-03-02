# Copyright (c) 2022 by Andreas Albrecht
# Copyright (c) 2012 - 2021, Udacity, Inc.
# All rights reserved.
# This file is part of the computer vision course project submission of Udacity's Self-driving
# Cars Nanodegree Program. It has been developed based on the starter code provided by Udacity
# on https://github.com/udacity/nd013-c1-vision-starter, and thus, is released under "Udacity
# License Agreement". Please see LICENSE file, which should be included as part of this package.

import argparse
import io
import os
import ray

import tensorflow.compat.v1 as tf
from PIL import Image
from psutil import cpu_count
from waymo_open_dataset import dataset_pb2 as open_dataset

from utils import get_module_logger, parse_frame, int64_feature, int64_list_feature, \
    bytes_list_feature, bytes_feature, float_list_feature


def create_tf_example(filename, source_id, encoded_jpeg, annotations, resize=True):
    """
    This function creates a tf.train.Example in object detection api format from a Waymo data frame.

    args:
        - filename [str]: name of the original tfrecord file
        - source_id [str]: original image source id (here: frame context name + camera name + frame index)
        - encoded_jpeg [bytes]: jpeg encoded image
        - annotations [protobuf object]: bboxes and classes

    returns:
        - tf_example [tf.Train.Example]: tf example in the objection detection api format.
    """
    if not resize:
        encoded_jpg_io = io.BytesIO(encoded_jpeg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size
        width_factor, height_factor = image.size
    else:
        image_tensor = tf.io.decode_jpeg(encoded_jpeg)
        height_factor, width_factor, _ = image_tensor.shape
        image_res = tf.cast(tf.image.resize(image_tensor, (640, 640)), tf.uint8)
        encoded_jpeg = tf.io.encode_jpeg(image_res).numpy()
        width, height = 640, 640

    mapping = {1: 'vehicle', 2: 'pedestrian', 4: 'cyclist'}
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    filename = filename.encode('utf8') # convert to bytes in utf8 format
    source_id = source_id.encode('utf8') # convert to bytes in utf8 format

    for ann in annotations:
        xmin, ymin = ann.box.center_x - 0.5 * ann.box.length, ann.box.center_y - 0.5 * ann.box.width
        xmax, ymax = ann.box.center_x + 0.5 * ann.box.length, ann.box.center_y + 0.5 * ann.box.width
        xmins.append(xmin / width_factor)
        xmaxs.append(xmax / width_factor)
        ymins.append(ymin / height_factor)
        ymaxs.append(ymax / height_factor)
        classes.append(ann.type)
        classes_text.append(mapping[ann.type].encode('utf8'))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(source_id),
        'image/encoded': bytes_feature(encoded_jpeg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example


def process_tfr(input_path, data_dir, extract_1_of_N=1):
    """
    Process a Waymo tfrecord file by extracting only every N-th frame of the front video camera
    images and labels and converting the extracted data items into a tfrecord file using tf object 
    detection api format. The processed tfrecord files are stored in a sub-directory "processed",
    which is created in the given data directory.

    args:
        - input_path [str]: local path to the downloaded Waymo tfrecord file
        - data_dir [str]: path to the data directory
        - extract_1_of_N [int]: extract 1 of every N image frames, defaults to 1

    returns:
        - output_path [str]: local path to the processed tfrecord file
        - num_of_images [int]: number of extracted images
    """
    # init number of images extracted from the original tfrecord file
    num_of_extracted_images = 0

    # camera name
    video_camera = 'FRONT'

    # get the current filename from local path
    filename = os.path.basename(input_path)
    if not os.path.exists(filename):
        logger.error(f'File does not exist {filename}')

    # create a sub-directory "processed" in the data directory to store the processed tfrecord files
    dest_dir = os.path.join(data_dir, 'processed')
    os.makedirs(dest_dir, exist_ok=True)

    # process current tfrecord file extracting only the front camera images and labels
    logger.info(f'Processing {input_path}')
    writer = tf.python_io.TFRecordWriter(f'{dest_dir}/{filename}')
    dataset = tf.data.TFRecordDataset(input_path, compression_type='')
    for idx, data in enumerate(dataset):
        # Sub-sequent image frames may be very similar if the time gap inbetween is small.
        # In order to reduce the number of very similar images, only every Nth image is
        # extracted and stored. Default: Extract and save every image frame
        if idx % extract_1_of_N == 0:
            # load and parse next frame
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy())) 
            encoded_jpeg, annotations = parse_frame(frame, camera_name=video_camera)
            # generate unique source id by combining frame context name, camera name and current frame index
            source_id = '{}_{}_{:03d}'.format(frame.context.name, video_camera, idx)
            # create and write tf.train.Example using tf object detection api format
            tf_example = create_tf_example(filename, source_id, encoded_jpeg, annotations)
            writer.write(tf_example.SerializeToString())
            # increment number of extracted images
            num_of_extracted_images += 1
    writer.close()

    # return output path and number of extracted images in the processed tfrecord file
    output_path = os.path.join(dest_dir, filename)
    return output_path, num_of_extracted_images


# remote function for distributed computation
@ray.remote  # uncomment decorator for parallel processing using ray
def process_tfrecord_file(local_path, data_dir, extract_1_of_N=1):
    """
    Process a Waymo tfrecord files by extracting 1 of every N image frames of the front video camera
    including the labels and converting the data into tfrecord files in tf object detection api format.

    args:
        - local_path [str]: local file path to the raw Waymo tfrecord file to be processed
        - data_dir [str]: local data directory where the processed tfrecord files will be stored in a
          sub-directory 'processed' (if 'processed' does not exist, it will be created)
        - extract_1_of_N [int]: extract 1 of every N image frames, defaults to 1
    """
    # re-import the logger because of multiprocesing
    logger = get_module_logger(__name__)

    # create 'processed' folder if it does not exist and process the raw tfrecord file in local_path
    dest_path, num_of_extracted_images = process_tfr(local_path, data_dir, extract_1_of_N)
    if not os.path.exists(dest_path):
        logger.error(f'tfrecord file was successfully processed and stored in {dest_path}')
    else:
        logger.info(f'Number of images extracted from {local_path}: {num_of_extracted_images}')
        logger.info(f'Processed tfrecord file stored in {dest_path}')


# Pocess selected tf rcord files
if __name__ == "__main__":
    """
    Example function calls:
    python process_tfrecords.py --data_dir ./data/waymo --source_dir ./data/waymo/raw --file_list tfrecord_files_train_val.txt --extract_1_of_N 10
    python process_tfrecords.py --data_dir ./data/waymo --source_dir ./data/waymo/raw --file_list tfrecord_files_test.txt --extract_1_of_N 1
    """
    logger = get_module_logger(__name__)
    parser = argparse.ArgumentParser(description='Process selected raw tfrecord files')
    parser.add_argument('--data_dir', required=True, type=str,
                        help='data directory (a sub-directory "processed" with the processed tfrecords will be created)')
    parser.add_argument('--source_dir', required=True, type=str,
                        help='source directory (e.g. ./data/waymo/raw)')
    parser.add_argument('--file_list', required=False, default='filenames.txt', type=str,
                        help='optional: list of tfrecord files to be processed')
    parser.add_argument('--extract_1_of_N', required=False, default=1, type=int,
                        help='optional: extract only 1 of N image frame, defaults to 1')
    args = parser.parse_args()
    data_dir = args.data_dir
    source_dir = args.source_dir
    file_list = args.file_list
    extract_1_of_N = args.extract_1_of_N

    # get list of all rfrecord files in source directory
    logger.info(f'Process selected raw tfrecord files ...')
    if not os.path.isdir(source_dir):
        logger.error(f'Source directory does not exist {source_dir}')
    source_file_list = []
    for file in os.listdir(source_dir):
        if file.endswith(".tfrecord"):
            source_file_list.append(file)

    # open the filenames file to get the list of tfrecord files to be processed
    with open(file_list, 'r') as f:
        filenames = [os.path.basename(fn) for fn in f.read().splitlines()]
    logger.info(f'Process {len(filenames)} tfrecord files. Be patient, this will take a while.')

    # make sure that files from the file list that exist in the source directory are processed
    source_filenames = [fn for fn in source_file_list if fn in filenames]

    # init ray
    ray.init(num_cpus=cpu_count())
    # process tfrecord files in parallel using multiple CPU cores (workers)
    workers = [process_tfrecord_file.remote(os.path.join(source_dir, fn), data_dir, extract_1_of_N) for fn in source_filenames]
    _ = ray.get(workers)
    
    # serial process in case the parallel process using ray doesn't work in local environment due to cuda issues
    #for fn in source_filenames:
    #    process_tfrecord_file(os.path.join(source_dir, fn), data_dir, extract_1_of_N)
        

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
import subprocess

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


def download_tfr(source_path, data_dir):
    """
    Download a single tfrecord file from the given source path to a destination
    sub-directory "raw" created in the given data directory if it does not exist

    args:
        - source_path [str]: source filepath to the tfrecord file
        - data_dir [str]: path to the data directory

    returns:
        - local_path [str]: path where the downloaded tfrecord file is saved
    """
    # create a destination sub-directory "raw" in the data directory downloading the tfrecord files
    dest_dir = os.path.join(data_dir, 'raw')
    os.makedirs(dest_dir, exist_ok=True)

    # download the tfrecord files from source
    cmd = ['gsutil', 'cp', source_path, f'{dest_dir}']
    logger.info(f'Downloading {source_path}')
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        logger.error(f'Could not download file {source_path}')

    # return local path to the downloaded tfrecord file
    local_path = os.path.join(dest_dir, os.path.basename(source_path))
    return local_path


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
def download_process_and_clean(source_path, data_dir, extract_1_of_N=1, cleanup=True):
    """
    Download and process a set of Waymo tfrecord files by extracting 1 of every N image
    frames of the front video camera including the labels and converting the data into
    tfrecord files in tf object detection api format. If cleanup is set to True the
    downloaded the raw tfrecord files are removed to save memory.

    args:
        - source_path [str]: path to the Waymo tfrecord file to be downloaded from google cloud
        - data_dir [str]: local data directory with a sub-directory 'raw' where the downlaoded 
          tfrecord files are stored and a sub-directory 'processed' where the processed tfrecord
          files are stored (if 'processed' does not exist, it will be created) 
        - extract_1_of_N [int]: extract 1 of every N image frames, defaults to 1
        - cleanup [bool]: remove the orignial tfrecord files to save space if True, defaults to True
    """
    # re-import the logger because of multiprocesing
    logger = get_module_logger(__name__)
    # create data directory to store the tfrecord files if it does not exist yet
    os.makedirs(data_dir, exist_ok=True)
    # download the tfrecord file from source path
    local_path = download_tfr(source_path, data_dir)
    if not os.path.exists(local_path):
        logger.error(f'Raw tfrecord file was successfully downloaded to {local_path}')
    else:
        logger.info(f'Raw tfrecord successfully downloaded to {local_path}')
    # create 'processed' folder if it does not exist and process the raw tfrecord file in local_path
    dest_path, num_of_extracted_images = process_tfr(local_path, data_dir, extract_1_of_N)
    if not os.path.exists(dest_path):
        logger.error(f'tfrecord file was successfully processed and stored in {dest_path}')
    else:
        logger.info(f'Number of images extracted from {local_path}: {num_of_extracted_images}')
        logger.info(f'Processed tfrecord file stored in {dest_path}')
    # remove the original tfrecord to save space
    if cleanup:
        logger.info(f'Deleting {local_path}')
        os.remove(local_path)
    else:
        logger.info(f'Original tfrecord kept under {local_path}')


if __name__ == "__main__":
    """
    Example function calls:
    python download_and_process_tfrecords.py --data_dir data/waymo/
    python download_and_process_tfrecords.py --data_dir data/waymo/ --file_list filenames.txt --size 100 --extract_1_of_N 1
    python download_and_process_tfrecords.py --data_dir data/waymo/ --file_list filenames.txt --size 100 --extract_1_of_N 10
    python download_and_process_tfrecords.py --data_dir data/waymo/ --file_list filenames.txt --size 100 --extract_1_of_N 10 --cleanup False
    """
    logger = get_module_logger(__name__)
    parser = argparse.ArgumentParser(description='Download and process tfrecord files')
    parser.add_argument('--data_dir', required=True, type=str,
                        help='data directory')
    parser.add_argument('--file_list', required=False, default='filenames.txt', type=str,
                        help='optional: list of tfrecord files to be downloaded and processed')
    parser.add_argument('--size', required=False, default=100, type=int,
                        help='optional: number of files from the list (top down) to be downloaded and processed, defaults to 100')
    parser.add_argument('--extract_1_of_N', required=False, default=1, type=int,
                        help='optional: extract only 1 of N image frame, defaults to 1')
    parser.add_argument("--cleanup", required=False, type=str, choices=['True', 'true', 'False', 'false'], default='True',
                        help="optional: clean up raw tfrecord files if True, defaults to True")
    args = parser.parse_args()
    data_dir = args.data_dir
    source_file_list = args.file_list
    size = args.size
    extract_1_of_N = args.extract_1_of_N
    clean_up = (args.cleanup.lower() == 'true')  # convert to boolean

    # open the filenames file to get the list of tfrecord files to be downloaded
    logger.info(f'Download and process tfrecord files from source ...')
    with open(source_file_list, 'r') as f:
        filenames = f.read().splitlines()
    logger.info(f'Number of files in file list: {len(filenames)}')
    if size > len(filenames):
        # limit size to length of file list
        size = len(filenames)
        logger.info(f'Reduce --size to length of file list: --size = {size}')
    logger.info(f'Download the first {len(filenames[:size])} files from file list. Be patient, this might take a long time.')

    # init ray
    ray.init(num_cpus=cpu_count())
    # download and process tfrecord files in parallel using multiple CPU cores (workers)
    workers = [download_process_and_clean.remote(fn, data_dir, extract_1_of_N, clean_up) for fn in filenames[:size]]
    _ = ray.get(workers)

    # use serial processing if parallel processing using ray causes cude issues in local environment
    #for fn in filenames[:size]:
    #    download_process_and_clean(fn, data_dir, extract_1_of_N, clean_up)

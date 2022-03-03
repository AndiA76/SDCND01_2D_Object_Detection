# Copyright (c) 2022 by Andreas Albrecht
# Copyright (c) 2012 - 2021, Udacity, Inc.
# All rights reserved.
# This file is part of the computer vision course project submission of Udacity's Self-driving
# Cars Nanodegree Program. It has been developed based on the starter code provided by Udacity
# on https://github.com/udacity/nd013-c1-vision-starter, and thus, is released under "Udacity
# License Agreement". Please see LICENSE file, which should be included as part of this package.

import argparse
import os
import ray
import subprocess

import tensorflow.compat.v1 as tf
from PIL import Image
from psutil import cpu_count
from waymo_open_dataset import dataset_pb2 as open_dataset

from utils import get_module_logger, parse_frame, int64_feature, int64_list_feature, \
    bytes_list_feature, bytes_feature, float_list_feature


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


# remote function for distributed computation
@ray.remote  # uncomment decorator for parallel processing using ray
def download(source_path, data_dir):
    """
    Download a set of Waymo tfrecord files to a sub-directory 'raw', which is created in the
    specified data directory.

    args:
        - source_path [str]: path to the Waymo tfrecord file to be downloaded from google cloud
        - data_dir [str]: local data directory with a sub-directory 'raw' where the downlaoded 
          tfrecord files are stored (if 'raw' does not exist, it will be created) 
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


if __name__ == "__main__":
    """
    Example function calls:
    python download_tfrecords.py --data_dir data/waymo/
    python download_tfrecords.py --data_dir data/waymo/ --file_list filenames.txt --size 100
    """
    logger = get_module_logger(__name__)
    parser = argparse.ArgumentParser(description='Download tfrecord files')
    parser.add_argument('--data_dir', required=True, type=str,
                        help='data directory')
    parser.add_argument('--file_list', required=False, default='filenames.txt', type=str,
                        help='optional: list of tfrecord files to be downloaded and processed')
    parser.add_argument('--size', required=False, default=100, type=int,
                        help='optional: number of files from the list (top down) to be downloaded and processed, defaults to 100')
    args = parser.parse_args()
    data_dir = args.data_dir
    source_file_list = args.file_list
    size = args.size

    # open the filenames file to get the list of tfrecord files to be downloaded
    logger.info(f'Download tfrecord files from source ...')
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
    # download tfrecord files in parallel using multiple CPU cores (workers)
    workers = [download.remote(fn, data_dir) for fn in filenames[:size]]
    _ = ray.get(workers)
    
    # use serial processing if parallel processing using ray causes cude issues in local environment
    #for fn in filenames[:size]:
    #    download(fn, data_dir)

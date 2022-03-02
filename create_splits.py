# Copyright (c) 2022 by Andreas Albrecht
# Copyright (c) 2012 - 2021, Udacity, Inc.
# All rights reserved.
# This file is part of the computer vision course project submission of Udacity's Self-driving
# Cars Nanodegree Program. It has been developed based on the starter code provided by Udacity
# on https://github.com/udacity/nd013-c1-vision-starter, and thus, is released under "Udacity
# License Agreement". Please see LICENSE file, which should be included as part of this package.

import argparse
import glob
import os
import shutil

import numpy as np

from utils import get_module_logger


def clean_target_dir(target_dir):
    """
    Remove all files, links and folders in an existing target directory.

    args:
        - target_dir [str]: existing target directory
    """
    logger.info('Clean target directory: {}'.format(target_dir))
    if not os.path.isdir(target_dir):
        logger.info('Error: Target directory to be cleaned does not exist!')
        return
    else:
        # check if target directory is empty
        for file in os.listdir(target_dir):
            filepath = os.path.join(target_dir, file)
            # delete all files and subdirectories if not empty
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
                elif os.path.islink(filepath):
                    os.unlink(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
            except Exception as ex:
                logger.info(
                    'Error: Failed to delete {}! Cause: {}'.format(filepath, ex)
                )

def split_data(source_dir, target_dir, N_train=82, N_val=15, N_test=3, use_symlinks=True):
    """
    Create a train-val or a train-val-test split from the tfrecord files in the source
    directory either by creating symbolic links (default: use_symlinks=True) to the source
    files in a train and val sub-directory, resp. in a train, val and test sub-directory, 
    or by moving the tfrecord files to the corresponding sub-directories in the target
    directory (use_symlinks=False).
    If the target directory does not exist it will be created. If train, val or test
    sub-directories do not exist the are created, too. If they do exist all contents are
    cleaned before splitting the data set.
    The number of tfrecord files for the training, validation and evtl. the test set must
    be specified as integer numbers. If the number of test files is set to None no test set
    will be created.
    
    Remarks:
        - If the specified splits do not yield sub-sets with an integer number of files the split
        ratio will be adapted to the closest split ratio that yields integer numbers.

    args:
        - source_dir [str]: source directory, /mnt/data/waymo/training_and_validation
        - target_dir [str]: target data directory, /mnt/data/waymo
        - N_train [int]: number of tfrecord files in the training set, default = 82
        - N_val [int]: number of tfrecord files in the validation set, default = 15
        - N_test [int]: number of tfrecord files in the test set, default = 3 (optional)
    """
    # validate source directory
    if not os.path.isdir(source_dir):
        logger.info('Error: Source directory does not exist!')
        logger.info('Execution stopped!')
        return
    else:
        # get list of tfrecord files in source directory
        tfrecord_filelist = [
            tfrecord_file for tfrecord_file in glob.glob(f'{source_dir}/*.tfrecord')
        ]
        # check if source directory contains any tfrecord files
        if not tfrecord_filelist:
            logger.info('Error: Source directory does not contain any *.tfrecord files!')
            logger.info('Execution stopped!')
            return
        else:
            # get number of tfrecord files in source directory
            N_tfrecords = len(tfrecord_filelist)

    # create target directory if it does not exists
    if not os.path.isdir(target_dir):
        logger.info('Warning: Target directory does not exist!')
        logger.info('Target directory created:\t{}'.format(target_dir))
        # create target directory
        os.makedirs(target_dir, mode = 0o777, exist_ok = True)

    # create train, val and test data directory if they do not already exist
    train_dir = os.path.join(target_dir, 'train')
    if not os.path.isdir(train_dir):
        logger.info('Training data directory created:\t{}'.format(train_dir))
        os.makedirs(train_dir, mode = 0o777, exist_ok = True)
    else:
        # clean training data directory if not empty
        clean_target_dir(train_dir)
    val_dir = os.path.join(target_dir, 'val')
    if not os.path.isdir(val_dir):
        logger.info('Validation data directory created:\t{}'.format(val_dir))
        os.makedirs(val_dir, mode = 0o777, exist_ok = True)
    else:
        # clean validation data directory if not empty
        clean_target_dir(val_dir)
    if N_test is not None:
        test_dir = os.path.join(target_dir, 'test')
        if not os.path.isdir(test_dir):
            logger.info('Testing data directory created:\t{}'.format(test_dir))
            os.makedirs(test_dir, mode = 0o777, exist_ok = True)
        # clean test data directory if not empty
        clean_target_dir(test_dir)

    # shuffle tfrecord file list
    np.random.shuffle(tfrecord_filelist)

    # split files
    if N_test is not None:
        # get the new nominator of the desired split ratio
        N_nom = sum([N_train, N_val, N_test])
        # check if the sum of train, val and test files corresponds to the number of source files
        if N_nom != N_tfrecords:
            logger.info('Warning: sum(N_train, N_val, N_test) does not match the number of source files!')
            # calculate the desired split ratio (!= actual split ratio)
            desired_split_ratio = (N_train/N_nom, N_val/N_nom, N_test/N_nom)
            logger.info('desired split ratio (N_train/N, N_val/N, N_test/N) = {}'.format(desired_split_ratio))
            logger.info('desired split N_train : N_val : N_test = {} : {} : {}'.format(N_train, N_val, N_test))
            logger.info('desired split ratio != actual split ratio')
            # adapt the desired split to the actual number of available tfrecords
            N_train = int(desired_split_ratio[0] * N_tfrecords)
            N_val = int(desired_split_ratio[1] * N_tfrecords)
            N_test = N_tfrecords - N_train - N_val
        else:
            # calculate the desired split ratio (== actual split ratio)
            desired_split_ratio = (N_train/N_nom, N_val/N_nom, N_test/N_nom)
            logger.info('desired split ratio (N_train/N, N_val/N, N_test/N) = {}'.format(desired_split_ratio))
            logger.info('desired split N_train : N_val : N_test = {} : {} : {}'.format(N_train, N_val, N_test))
            logger.info('desired split ratio == actual split ratio')
        # calculate the actual split ratio
        split_ratio = (N_train/N_tfrecords, N_val/N_tfrecords, N_test/N_tfrecords)
        # split the tfrecord data set
        train_files, val_files, test_files = np.split(tfrecord_filelist, [N_train, N_train + N_val])
        logger.info('actual split ratio (N_train/N, N_val/N, N_test/N) = {}'.format(split_ratio))
        logger.info('actual split N_train : N_val : N_test = {} : {} : {}'.format(N_train, N_val, N_test))
    else:
        # get the new nominator of the desired split ratio
        N_nom = sum([N_train, N_val])
        # check if the sum of train and val files corresponds to the number of source files
        if N_nom != N_tfrecords:
            logger.info('Warning: sum(N_train, N_val) does not match the number of source files!')
            # calculate the desired split ratio (!= actual split ratio)
            desired_split_ratio = (N_train/N_nom, N_val/N_nom)
            logger.info('desired split ratio (N_train/N, N_val/N) = {}'.format(desired_split_ratio))
            logger.info('desired split N_train : N_val = {} : {}'.format(N_train, N_val))
            logger.info('desired split ratio != actual split ratio')
            # adapt the desired split to the actual number of available tfrecords
            N_train = int(desired_split_ratio[0] * N_tfrecords)
            N_val = N_tfrecords - N_train
        else:
            # calculate the desired split ratio (== actual split ratio)
            desired_split_ratio = (N_train/N_nom, N_val/N_nom)
            logger.info('desired split ratio (N_train/N, N_val/N) = {}'.format(desired_split_ratio))
            logger.info('desired split N_train : N_val = {} : {}'.format(N_train, N_val))
            logger.info('desired split ratio == actual split ratio')
        # calculate the actual split ratio
        split_ratio = (N_train/N_tfrecords, N_val/N_tfrecords)
        # split the tfrecord data set
        train_files, val_files = np.split(tfrecord_filelist, [N_train])
        logger.info('actual split ratio (N_train/N, N_val/N) = {}'.format(split_ratio))
        logger.info('actual split N_train : N_val = {} : {}'.format(N_train, N_val))

    # create symbolic links to the training files in the train directory and print sorted file list to log file
    sorted_train_files = sorted(train_files, reverse=False)
    logger.info('sorted train files (number of files = {}):'.format(len(sorted_train_files)))
    for tfrecord_file in sorted_train_files:
        logger.info('\t{} -> {}'.format(tfrecord_file, os.path.join(train_dir, os.path.split(tfrecord_file)[1])))
        if use_symlinks:
            # create symbolik links to tfrecords from training set in train_dir
            os.symlink(tfrecord_file, os.path.join(train_dir, os.path.split(tfrecord_file)[1]))
        else:
            # move tfrecord files from training set to train_dir
            shutil.move(tfrecord_file, train_dir)
    # create symbolic links to the validation files in the val directory and print sorted file list to log file
    sorted_val_files = sorted(val_files, reverse=False)
    logger.info('sorted val files (number of files = {}):'.format(len(sorted_val_files)))
    for tfrecord_file in sorted_val_files:
        logger.info('\t{} -> {}'.format(tfrecord_file, os.path.join(val_dir, os.path.split(tfrecord_file)[1])))
        if use_symlinks:
            # create symbolik links to tfrecords from validation set in val_dir
            os.symlink(tfrecord_file, os.path.join(val_dir, os.path.split(tfrecord_file)[1]))
        else:
            # move tfrecords from validation set to val_dir
            shutil.move(tfrecord_file, val_dir)
    if N_test is not None:
        # create symbolic links to the test files in the test directory and print sorted file list to log file
        sorted_test_files = sorted(test_files, reverse=False)
        logger.info('sorted test files (number of files = {}):'.format(len(sorted_test_files)))
        for tfrecord_file in sorted_test_files:
            logger.info('\t{} -> {}'.format(tfrecord_file, os.path.join(test_dir, os.path.split(tfrecord_file)[1])))
            if use_symlinks:
                # create symbolik links to tfrecords from test set in test_dir
                os.symlink(tfrecord_file, os.path.join(test_dir, os.path.split(tfrecord_file)[1]))
            else:
                # move tfrecord files from test set to test_dir
                shutil.move(tfrecord_file, test_dir)


if __name__ == "__main__":
    '''
    command line function call examples using symbolic links (use absolute path specification w.r.t. to docker root dir):
    python create_splits.py --source_dir /app/project/data/waymo/training_and_validation --target_dir /app/project/data/waymo --num_train_files 82 --num_val_files 15 --use_symlinks True
    python create_splits.py --source_dir /app/project/data/waymo/training_and_validation --target_dir /app/project/data/waymo
    python create_splits.py --source_dir /app/project/data/waymo/processed --target_dir /app/project/data/waymo --num_train_files 82 --num_val_files 15 --num_test_files 3

    command line function call examples moving files (relative path specifications are ok):
    python create_splits.py --source_dir ./data/waymo/training_and_validation --target_dir ./data/waymo --num_train_files 82 --num_val_files 15 --use_symlinks False

    Remarks:
        - Links to shared memory set from within the docker environment do not work outside the docker environment
        - Therefore, when running this script inside a docker container using symbolic links you should use the absolute
          paths with respect to the root directory in the docker environment. 
    '''
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source_dir', required=True,
                        help='source directory')
    parser.add_argument('--target_dir', required=True,
                        help='target directory')
    parser.add_argument("--num_train_files", required=False, type=int, default=82,
                        help="optional: number of tfrecord files for the training set (default = 82)")
    parser.add_argument("--num_val_files", required=False, type=int, default=15,
                        help="optional: number of tfrecord files for the validation set (default = 15)")
    parser.add_argument("--num_test_files", required=False, type=int,
                        help="optional: number of tfrecord files for the test set, e.g. 3")
    parser.add_argument("--use_symlinks", required=False, type=str, choices=['True', 'true', 'False', 'false'], default='True',
                        help="optional: use symbolic links to the original data sets if True, or move files if False")
    args = parser.parse_args()

    # get logger module and split the data set according to the given arguments
    logger = get_module_logger(__name__)
    if args.num_test_files:
        logger.info('Creating train-val-test splits ...')
        split_data(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            N_train=args.num_train_files,
            N_val=args.num_val_files,
            N_test=args.num_test_files,
            use_symlinks=(args.use_symlinks.lower()=='true'),  # convert to boolean
        )
    else:
        logger.info('Creating train-val splits ...')
        split_data(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            N_train=args.num_train_files,
            N_val=args.num_val_files,
            N_test=None,
            use_symlinks=(args.use_symlinks.lower()=='true'),  # convert to boolean
        )

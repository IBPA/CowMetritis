"""
Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:
    Utility functions used in the project.

To-do:
"""
# standard imports
import logging as log
import os
import pickle
from shutil import copyfile

# third party imports
import numpy as np


def dir_exists(directory):
    """
    Check if directory exists.

    Inputs:
        directory: (str) Directory to check.

    Returns:
        (bool) True if directory exists, False otherwise.
    """
    return os.path.isdir(directory)


def save_pkl(obj, save_to):
    """
    Pickle the object.

    Inputs:
        obj: (object) Object to pickle.
        save_to: (str) Filepath to pickle the object to.
    """
    with open(save_to, 'wb') as fid:
        pickle.dump(obj, fid)


def load_pkl(load_from):
    """
    Load the pickled object.

    Inputs:
        save_to: (str) Filepath to pickle the object to.

    Returns:
        (object) Loaded object.
    """
    with open(load_from, 'rb') as fid:
        obj = pickle.load(fid)

    return obj


def create_dir(directory):
    """
    Create directory only if directory does not exist already.
    If directory exists, raise RuntimeError.

    Inputs:
        directory: directory to create
    """
    if os.path.isdir(directory):
        raise RuntimeError('Directory \'{}\' already exists!'.format(directory))
    else:
        log.info('Creating directory: %s', directory)
        os.makedirs(directory)


def get_results_of_search(results, report_score_using='f1', scoring=['f1', 'average_precision'], count=5):
    if isinstance(scoring, str):
        scoring = ['score']
        report_score_using = 'score'

    for score in scoring:
        log.info('Scoring results of %s', score)

        for idx in range(1, count + 1):
            runs = np.flatnonzero(results['rank_test_{}'.format(score)] == idx)

            for run in runs:
                log.info('evaluation rank: {}'.format(idx))
                log.info('score: {}'.format(results['mean_test_{}'.format(score)][run]))
                log.info('std: {}'.format(results['std_test_{}'.format(score)][run]))
                log.info(results['params'][run])

            if (report_score_using == score) and (idx == 1):
                best_params = results['params'][run]
                best_score = results['mean_test_{}'.format(score)][run]

    return best_params, best_score


def str_is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def str_is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def check_str_type(s):
    if str_is_integer(s):
        return int
    elif str_is_float(s):
        return float
    else:
        return str


def create_backup(original, backup_extension='.bak'):
    backup_filename = original + backup_extension
    copyfile(original, backup_filename)


def backup_remove_original(original, backup_extension='.bak'):
    create_backup(original, backup_extension)
    os.remove(original)

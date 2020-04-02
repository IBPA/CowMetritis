"""
Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:
    Description goes here.

To-do:
"""
# standard imports
import argparse
import logging as log
import os
import sys

# local imports
from managers.preprocess_manager import PreprocessManager
from utils.config_parser import ConfigParser
from utils.set_logging import set_logging
from utils.visualization import visualize_missing_values, plot_projection

# global variables
DEFAULT_CONFIG_FILE = './config/main.ini'


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Description goes here.')

    parser.add_argument(
        '--config_file',
        default=DEFAULT_CONFIG_FILE,
        help='Path to the .ini configuration file.')

    return parser.parse_args()


def main():
    """
    Main function.
    """
    # set log, parse args, and read configuration
    set_logging()
    args = parse_argument()
    configparser = ConfigParser(args.config_file)

    # init object for preprocessing
    pmanager = PreprocessManager(
        configparser.get_str('preprocess_config'))

    # read data
    pd_data = pmanager.read_data()

    # visualize missing values
    visualize_missing_values(
        pd_data,
        configparser.get_str('visualization_dir'))

    # get independent / dependent variables
    X, y = pmanager.get_X_and_y(pd_data)

    # scale features
    X = pmanager.scale_features(X)

    # impute missing value
    X = pmanager.impute_missing_values(X)

    X_pc = pmanager.reduce_dimension(X, 'PCA')
    plot_projection(X_pc, y, configparser.get_str('visualization_dir'), 'PCA')

    X_tsne = pmanager.reduce_dimension(X, 'tSNE')
    plot_projection(X_tsne, y, configparser.get_str('visualization_dir'), 'TSNE')

    # detect and remove outliers
    outliers = pmanager.detect_outlier(X)


if __name__ == '__main__':
    main()

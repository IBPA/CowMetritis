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

    pmanager = PreprocessManager(
        configparser.getstr('input_data'))

if __name__ == '__main__':
    main()
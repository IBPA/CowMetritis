"""
Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Preprocess manager.

To-do:
"""
# standard imports
import logging as log
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

# third party imports
import numpy as np
import pandas as pd


class PreprocessManager:
    """
    Preprocess the input data.
    """

    def __init__(self, filepath):
        """
        Class initializer.

        Inputs:
        """
        pd_data = pd.read_csv(filepath, na_values='.')
        print(pd_data.head())

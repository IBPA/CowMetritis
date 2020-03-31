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
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

# third party imports
import numpy as np
import pandas as pd
from sklearn import preprocessing

# local imports
from utils.config_parser import ConfigParser
from missing_value_imputer import knn_imputer, iterative_imputer
from outlier_detector import isolation_forest, one_class_svm, local_outlier_factor


class PreprocessManager:
    """
    Preprocess the input data.
    """

    def __init__(self, configfile):
        """
        Class initializer.

        Inputs:
        """
        # load config parser
        configparser = ConfigParser(configfile)

        # variables
        self.input_file = configparser.get_str('input_data')
        self.string_cols = configparser.get_str_list('string_columns')
        self.category_cols = configparser.get_str_list('category_columns')
        self.dependent_cols = configparser.get_str_list('dependent_columns')
        self.independent_columns = configparser.get_str_list('independent_columns')
        self.mvi_mode = configparser.get_str('mvi_mode')
        self.outlier_mode = configparser.get_str('outlier_mode')
        self.label_lookup = {}

    def read_raw_data(self, encode=True):
        pd_data = pd.read_csv(self.input_file, na_values='.', index_col='ID')

        if encode:
            return self._encode_label(pd_data)
        else:
            return pd_data

    def _encode_label(self, pd_data):
        pd_encoded = pd_data.copy()

        for column in self.string_cols:
            le = preprocessing.LabelEncoder()
            le.fit(pd_encoded[column].tolist())

            log.info('Encoding label for column \'%s\': %s', column, list(le.classes_))
            pd_encoded[column] = le.transform(pd_encoded[column].tolist())
            self.label_lookup[column] = le

        return pd_encoded

    def decode_label(self, pd_data):
        pd_decoded = pd_data.copy()

        for column in self.string_cols:
            le = self.label_lookup[column]

            log.info('Decoding label for column \'%s\': %s', column, list(le.classes_))
            pd_decoded[column] = le.inverse_transform(pd_decoded[column].astype(int).tolist())

        return pd_decoded

    def impute_missing_values(self, pd_data):
        pd_dependent = pd_data[self.dependent_cols]
        pd_independent = pd_data.drop(self.dependent_cols, axis=1)

        if self.mvi_mode == 'knn':
            pd_imputed = knn_imputer(pd_independent)
        elif self.mvi_mode == 'iterative':
            pd_imputed = iterative_imputer(pd_independent)
        else:
            raise ValueError('Invalid MVI mode: {}'.format(self.mvi_mode))

        pd_imputed = pd.concat([pd_dependent, pd_imputed], axis=1)
        pd_imputed = pd_imputed[pd_data.columns]

        imputed_columns = pd_independent.columns[pd_independent.isna().any()].tolist()
        imputed_category_columns = [col for col in imputed_columns if col in self.category_cols]

        for column in imputed_category_columns:
            pd_imputed[column] = pd_imputed[column].apply(lambda x: round(x))

        return pd_imputed

    def detect_outlier(self, pd_data):
        pd_independent = pd_data[self.independent_columns]

        if self.outlier_mode == 'isolation_forest':
            outliers = isolation_forest(pd_independent)
        elif self.outlier_mode == 'one_class_svm':
            outliers = one_class_svm(pd_independent)
        elif self.outlier_mode == 'LOF':
            outliers = local_outlier_factor(pd_independent)
        else:
            raise ValueError('Invalid outlier mode: {}'.format(self.outlier_mode))


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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
        self.independent_cols = configparser.get_str_list('independent_columns')
        self.mvi_mode = configparser.get_str('mvi_mode')
        self.outlier_mode = configparser.get_str('outlier_mode')
        self.projection_dim = configparser.get_int('projection_dimension')
        self.label_lookup = {}

    def _assign_label(self, row):
        if row['TRT'] == 'CEF' and row['Cured'] == 1:
            return 'treated_cured'
        elif row['TRT'] == 'CEF' and row['Cured'] == 0:
            return 'treated_uncured'
        elif row['TRT'] == 'CON' and row['Cured'] == 1:
            return 'untreated_cured'
        elif row['TRT'] == 'CON' and row['Cured'] == 0:
            return 'untreated_uncured'
        else:
            raise ValueError('Cannot assign label to row: {}'.format(row))

    def read_data(self, encode=True):
        pd_data = pd.read_csv(self.input_file, na_values='.', index_col='ID')

        if encode:
            return self.encode_label(pd_data)
        else:
            return pd_data

    def get_X_and_y(self, pd_data):
        pd_decoded = self.decode_label(pd_data)

        X = pd_data[self.independent_cols]
        y = pd_decoded.apply(lambda row: self._assign_label(row), axis=1)

        return X, y

    def encode_label(self, pd_data):
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
        if self.mvi_mode == 'knn':
            pd_imputed = knn_imputer(pd_data)
        elif self.mvi_mode == 'iterative':
            pd_imputed = iterative_imputer(pd_data)
        else:
            raise ValueError('Invalid MVI mode: {}'.format(self.mvi_mode))

        imputed_columns = pd_data.columns[pd_data.isna().any()].tolist()
        imputed_category_columns = [col for col in imputed_columns if col in self.category_cols]

        for column in imputed_category_columns:
            pd_imputed[column] = pd_imputed[column].apply(lambda x: round(x))

        return pd_imputed

    def get_principal_components(self, X):
        pca = PCA(n_components=self.projection_dim)
        principal_components = pca.fit_transform(X)

        pd_new_X = pd.DataFrame(
            principal_components,
            index=X.index,
            columns=['pc' + str(i+1) for i in range(self.projection_dim)])

        return pd_new_X

    def get_tsne(self, X):
        tsne = TSNE(n_components=self.projection_dim)
        embeddings = tsne.fit_transform(X)

        pd_new_X = pd.DataFrame(
            embeddings,
            index=X.index,
            columns=['embedding' + str(i+1) for i in range(self.projection_dim)])

        return pd_new_X

    def detect_outlier(self, pd_data):
        if self.outlier_mode == 'isolation_forest':
            outliers = isolation_forest(pd_data)
        elif self.outlier_mode == 'one_class_svm':
            outliers = one_class_svm(pd_data)
        elif self.outlier_mode == 'LOF':
            outliers = local_outlier_factor(pd_data)
        else:
            raise ValueError('Invalid outlier mode: {}'.format(self.outlier_mode))

        return outliers

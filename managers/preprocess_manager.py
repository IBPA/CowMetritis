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
from sklearn.decomposition import PCA, SparsePCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import pearsonr

# local imports
from missing_value_imputer import knn_imputer, iterative_imputer, missforest_imputer
from outlier_detector import isolation_forest, local_outlier_factor
from utils.visualization import plot_pairwise_corr
from utils.utilities import save_pkl, load_pkl


class PreprocessManager:
    """
    Preprocess the input data.
    """

    def __init__(self, configparser, classifier='DEFAULT'):
        """
        Class initializer.

        Inputs:
            configfile: (str) Configuration file path.
        """
        # read the configuration file
        self.input_file = configparser.get_str('input_data')
        self.independent_cols = configparser.get_str_list('independent_columns')
        self.dependent_col = configparser.get_str('dependent_column')
        self.category_cols = configparser.get_str_list('category_columns')
        self.scale_mode = configparser.get_str('scale_mode', section=classifier)
        self.mvi_mode = configparser.get_str('mvi_mode', section=classifier)
        self.outlier_mode = configparser.get_str('outlier_mode', section=classifier)
        self.dimension_reduction_mode = configparser.get_str('dimension_reduction_mode')
        self.projection_dim = configparser.get_int('projection_dimension')
        self.random_state = configparser.get_int('random_state')

        if configparser.get_str('model_dir'):
            self.model_path = os.path.join(configparser.get_str('model_dir'), classifier)
        else:
            self.model_path = None

        # initialize label lookup dictionary
        self.category_lookup = {}

    def read_data(self):
        """
        Read data.

        Inputs:
            encode: (bool, optional) If True, encode the categorical columns.

        Returns:
            (DataFrame) Read data.
        """
        return pd.read_csv(self.input_file, na_values='.', index_col='ID')

    def get_X_and_y(self, pd_data):
        """
        Extract independent and dependent variables.

        Inputs:
            pd_data: (DataFrame) Input raw data.

        Returns:
            X: (DataFrame) Independent variables.
            y: (DataFrame) Dependent variables.
        """
        X = pd_data[self.independent_cols]
        y = pd_data[self.dependent_col]

        return X, y

    def get_X(self, pd_data):
        """
        Extract independent variables.

        Inputs:
            pd_data: (DataFrame) Input raw data.

        Returns:
            (DataFrame) Independent variables.
        """
        return pd_data[self.independent_cols]

    def scale_features(self, X, final_model=False, pkl_filename='scaler.pkl'):
        """
        Scale the independent variables.

        Inputs:
            X: (DataFrame) Independent variables.

        Returns:
            pd_new_X: (DataFrame) Scaled independent variables.
        """
        if final_model:
            scaler = load_pkl(os.path.join(self.model_path, pkl_filename))
        else:
            if self.scale_mode.lower() == 'standard':
                scaler = StandardScaler().fit(X)
            elif self.scale_mode.lower() == 'minmax':
                scaler = MinMaxScaler().fit(X)
            elif self.scale_mode.lower() == 'robust':
                scaler = RobustScaler().fit(X)
            else:
                raise ValueError('Invalid scaling mode: {}'.format(self.scale_mode))

            if self.model_path:
                save_pkl(scaler, os.path.join(self.model_path, pkl_filename))

        pd_new_X = pd.DataFrame(
            scaler.transform(X),
            index=X.index,
            columns=X.columns)

        return pd_new_X

    def impute_missing_values(self, X, round_categories=False):
        """
        Inpute missing values of the independent variables.

        Inputs:
            X: (DataFrame) Independent variables.
            round_categories: (bool, optional) If True, round the
                categorical columns to the nearest integer.

        Returns:
            pd_new_X: (DataFrame) Independent variables with
                missinv values filled.
        """
        if self.mvi_mode.lower() == 'knn':
            pd_imputed = knn_imputer(X)
        elif self.mvi_mode.lower() == 'iterative':
            pd_imputed = iterative_imputer(X, self.random_state)
        elif self.mvi_mode.lower() == 'missforest':
            pd_imputed = missforest_imputer(X, self.random_state)
        else:
            raise ValueError('Invalid MVI mode: {}'.format(self.mvi_mode))

        if round_categories:
            imputed_columns = X.columns[X.isna().any()].tolist()
            imputed_category_columns = [col for col in imputed_columns if col in self.category_cols]

            for column in imputed_category_columns:
                pd_imputed[column] = pd_imputed[column].apply(lambda x: round(x))

        return pd_imputed

    def reduce_dimension(self, X):
        """
        Perform dimensionality reduction.

        Inputs:
            X: (DataFrame) Independent variables.

        Returns:
            pd_new_X: (DataFrame) Reduced dimension independent variables.
            mode: (str) Dimensionality reduction used (PCA | tSNE)
        """
        if self.dimension_reduction_mode.lower() == 'pca':
            model = PCA(n_components=self.projection_dim)
            column_prefix = 'pc'
        elif self.dimension_reduction_mode.lower() == 'sparsepca':
            model = SparsePCA(n_components=self.projection_dim)
            column_prefix = 'pc'
        elif self.dimension_reduction_mode.lower() == 'tsne':
            model = TSNE(n_components=self.projection_dim)
            column_prefix = 'embedding'
        else:
            raise ValueError('Invalid mode: {}'.format(self.dimension_reduction_mode))

        pd_new_X = pd.DataFrame(
            model.fit_transform(X),
            index=X.index,
            columns=[column_prefix + str(i+1) for i in range(self.projection_dim)])

        return pd_new_X, self.dimension_reduction_mode

    def detect_outlier(self, X):
        """
        Detect outliers.

        Inputs:
            X: (DataFrame) Independent variables.

        Returns:
            index: (list) False for outliers and True for inliers.
        """
        if self.outlier_mode.lower() == 'isolation_forest':
            index = isolation_forest(X, self.random_state)
        elif self.outlier_mode.lower() == 'lof':
            index = local_outlier_factor(X)
        else:
            raise ValueError('Invalid outlier mode: {}'.format(self.outlier_mode))

        return index

    def remove_outlier(self, X, y, index):
        """
        Remove outliers.

        Inputs:
            X: (DataFrame) Independent variables.
            y: (DataFrame) Dedependent variables.
            index: (list) False for outliers and True for inliers.

        Returns:
            X_inliers: (DataFrame) Independent variables without outliers.
            y_inliers: (DataFrame) Dedependent variables without outliers.
        """
        X_index = X.index.tolist()
        inliers = [X_index[i] for i, is_outlier in enumerate(index) if is_outlier]

        X_inliers = X.loc[inliers]
        y_inliers = y.loc[inliers]

        return X_inliers, y_inliers

    def feature_analysis(self, X, y, save_to=None):
        """
        Do feature analysis like Pearson correlation.

        Inputs:
            X: (DataFrame) Features.
            y: (DataFrame) Dependent variable.
            save_to: (str) If not None, save the pairwise correlation figure to here.
        """
        if save_to:
            plot_pairwise_corr(pd.concat([X, y], axis=1).corr(), save_to)

        # pairwise pearson correlation
        pearson_result = [pearsonr(X[feature], y) for feature in list(X)]

        scores = list(zip(*pearson_result))[0]
        scores_abs = [abs(s) for s in scores]
        pvalues = list(zip(*pearson_result))[1]

        indices = np.argsort(scores_abs)[::-1]
        ranking = [list(X)[idx] for idx in indices]

        # log the feature ranking
        log.debug('Pairwise feature correlation ranking:')
        for f in range(X.shape[1]):
            log.debug('%d. %s: %f (%f)', f+1, ranking[f], scores[indices[f]], pvalues[indices[f]])

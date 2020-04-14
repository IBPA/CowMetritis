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
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA, SparsePCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

# local imports
from missing_value_imputer import knn_imputer, iterative_imputer, missforest_imputer
from outlier_detector import isolation_forest, one_class_svm, local_outlier_factor
from utils.visualization import plot


class PreprocessManager:
    """
    Preprocess the input data.
    """

    def __init__(self, configparser):
        """
        Class initializer.

        Inputs:
            configfile: (str) Configuration file path.
        """
        # read the configuration file
        self.input_file = configparser.get_str('input_data')
        self.independent_cols = configparser.get_str_list('independent_columns')
        self.dependent_col = configparser.get_str('dependent_column')
        self.string_cols = configparser.get_str_list('string_columns')
        self.category_cols = configparser.get_str_list('category_columns')
        self.scale_mode = configparser.get_str('scale_mode')
        self.mvi_mode = configparser.get_str('mvi_mode')
        self.outlier_mode = configparser.get_str('outlier_mode')
        self.dimension_reduction_mode = configparser.get_str('dimension_reduction_mode')
        self.projection_dim = configparser.get_int('projection_dimension')
        self.rfe_classifier = configparser.get_str('rfe_classifier')
        self.random_state = configparser.get_int('random_state')

        # initialize label lookup dictionary
        self.category_lookup = {}

    def encode_category(self, pd_data):
        """
        Encode the categorical columns of the input data.

        Inputs:
            pd_data: (DataFrame) Data that needs category encoding.

        Returns:
            pd_encoded: (DataFrame) Data with categories encoded.
        """
        pd_encoded = pd_data.copy()

        # we only encode columns with string data.
        # integer categrical columns are not encoded.
        if not self.string_cols:
            log.info('No columns to encode.')
            return pd_data

        for column in self.string_cols:
            le = preprocessing.LabelEncoder()
            le.fit(pd_encoded[column].tolist())

            log.info('Encoding label for column \'%s\': %s', column, list(le.classes_))
            pd_encoded[column] = le.transform(pd_encoded[column].tolist())

            # store the encoder as dictionary for future decoding
            self.category_lookup[column] = le

        return pd_encoded

    def decode_category(self, pd_data):
        """
        Decode the categorical columns of the input data.

        Inputs:
            pd_data: (DataFrame) Data that needs category decoding.

        Returns:
            pd_decoded: (DataFrame) Data with categories decoded.
        """
        pd_decoded = pd_data.copy()

        # we only decode columns with string data.
        # integer categrical columns are not decoded.
        if not self.string_cols:
            log.info('No columns to decode.')
            return pd_data

        for column in self.string_cols:
            le = self.category_lookup[column]

            log.info('Decoding label for column \'%s\': %s', column, list(le.classes_))
            pd_decoded[column] = le.inverse_transform(pd_decoded[column].astype(int).tolist())

        return pd_decoded

    def read_data(self, encode=True):
        """
        Read data.

        Inputs:
            encode: (bool, optional) If True, encode the categorical columns.

        Returns:
            pd_data: (DataFrame) Read data.
        """
        pd_data = pd.read_csv(self.input_file, na_values='.', index_col='ID')

        if encode:
            return self.encode_category(pd_data)
        else:
            return pd_data

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

    def scale_features(self, X):
        """
        Scale the independent variables.

        Inputs:
            X: (DataFrame) Independent variables.

        Returns:
            pd_new_X: (DataFrame) Scaled independent variables.
        """
        if self.scale_mode.lower() == 'standard':
            scaler = StandardScaler()
        elif self.scale_mode.lower() == 'minmax':
            scaler = MinMaxScaler()
        elif self.scale_mode.lower() == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError('Invalid scaling mode: {}'.format(self.scale_mode))

        pd_new_X = pd.DataFrame(
            scaler.fit_transform(X),
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
        elif self.outlier_mode.lower() == 'one_class_svm':
            index = one_class_svm(X)
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

    def feature_selection(self, X, y, save_to=None):
        if self.rfe_classifier.lower() == 'randomforestclassifier':
            clf = RandomForestClassifier(random_state=self.random_state)
        else:
            raise ValueError('Invalid classifier: {}'.format(self.rfe_classifier))

        selector = RFECV(clf, step=1, min_features_to_select=1, scoring='f1', n_jobs=-1)
        selector = selector.fit(X, y)

        features = X.columns.to_numpy()
        selected_features = features[selector.support_]
        dropped_features = features[~selector.support_]

        log.info('Number of features selected: %d', selector.n_features_)
        log.info('Selected features: %s', selected_features)
        log.info('Dropped features: %s', dropped_features)
        log.info('Feature ranking: %s', selector.ranking_)

        if save_to:
            plot(
                range(1, len(selector.grid_scores_) + 1),
                selector.grid_scores_,
                save_to,
                'feature_selection.png')

        return X[selected_features.tolist()]

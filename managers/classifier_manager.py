"""
Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Classifier manager.

To-do:
"""
# standard imports
import ast
import logging as log
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

# third party imports
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# local imports
from utils.utilities import check_str_type
from utils.visualization import plot_sfs


class ClassifierManager:
    """
    All about the classifiers.
    """

    def __init__(self, configparser):
        """
        Class initializer.

        Inputs:
            configfile: (str) Configuration file path.
        """
        # load config parser
        self.configparser = configparser

        # parse DEFAULT config
        self.visualization_dir = self.configparser.get_str('visualization_dir')
        self.classifier = self.configparser.get_str('classifier')

        # parse mode (gridsearch or normal) and get parameters
        self.mode, self.parameters = self._parse_param()

        # assign appropriate classifier
        self.clf = self._build_pipeline()

    def _build_pipeline(self):
        """
        Built the classifier pipeline.

        Returns:
            clf: (Pipeline) Pipeline.
        """
        # assign appropriate classifier
        if self.classifier.lower() == 'dummyclassifier':
            clf = DummyClassifier(strategy='most_frequent')
        elif self.classifier.lower() == 'decisiontreeclassifier':
            clf = DecisionTreeClassifier(**self.parameters if not self.mode == 'grid' else {})
        elif self.classifier.lower() == 'gaussiannb':
            clf = GaussianNB()
        elif self.classifier.lower() == 'multinomialnb':
            clf = MultinomialNB()
        elif self.classifier.lower() == 'svc':
            clf = SVC(probability=True, **self.parameters)
        elif self.classifier.lower() == 'adaboostclassifier':
            clf = AdaBoostClassifier(**self.parameters if not self.mode == 'grid' else {})
        elif self.classifier.lower() == 'randomforestclassifier':
            clf = RandomForestClassifier(
                n_jobs=-1, **self.parameters if not self.mode == 'grid' else {})
        elif self.classifier.lower() == 'mlpclassifier':
            clf = MLPClassifier(max_iter=3000, **self.parameters if not self.mode == 'grid' else {})
        else:
            raise ValueError('Invalid classifier: {}'.format(self.classifier))

        log.info('Selected classifier: %s', self.classifier)
        log.debug('Classifier info: %s', clf)

        # SMOTE over-sample
        smote = SMOTE(sampling_strategy='minority')
        clf = Pipeline([('SMOTE', smote), (self.classifier, clf)])

        return clf

    def _parse_param(self):
        """
        Read the configuration file and parse the parameters
        to be used by the classifiers.

        Returns:
            mode: (str) Either one of (normal, grid).
            parameters: (dict) Dictionary where key is the parameter
                name and value is the actual parameter value to use.
        """
        sections = self.configparser.sections()
        grid_search_section = '{}_GridSearch'.format(self.classifier)
        best_result_section = '{}_Best'.format(self.classifier)

        parameters = {}

        if best_result_section in sections:
            mode = 'normal'

            params_dict = self.configparser.get_section_as_dict(section=best_result_section)
            del params_dict['classifier']
            del params_dict['visualization_dir']

            for key, value in params_dict.items():
                dtype = check_str_type(value[0])

                # do special parsing for MLPClassifier
                if key == 'hidden_layer_sizes':
                    hidden_layer_sizes = self.configparser.get_str(
                        'hidden_layer_sizes', 'MLPClassifier_Best')

                    parameters[key] = ast.literal_eval(hidden_layer_sizes)
                    continue

                if dtype == str:
                    parameters[key] = value[0]
                elif dtype == int:
                    parameters[key] = int(value[0])
                elif dtype == float:
                    parameters[key] = float(value[0])
                else:
                    raise ValueError('Invalid dtype: {}'.format(dtype))
        elif grid_search_section in sections:
            mode = 'grid'

            params_dict = self.configparser.get_section_as_dict(section=grid_search_section)
            del params_dict['classifier']
            del params_dict['visualization_dir']

            # get parameter names
            keys = params_dict.keys()
            params_range = []
            params_list = list(keys)

            for key in keys:
                for suffix in ['_start', '_end', '_increment']:
                    if suffix in key:
                        params_range.append(key.replace(suffix, ''))
                        params_list.remove(key)

            params_range = list(set(params_range))

            log.debug('Parameters using range: %s', params_range)
            log.debug('Parameters using list: %s', params_list)

            # parse parameter values
            for param in params_list:
                parameters[param] = params_dict[param]

            for param in params_range:
                mapping = check_str_type(params_dict['{}_start'.format(param)][0])

                if mapping == str:
                    raise ValueError('Invalid dtype: {}'.format(mapping))

                start = mapping(params_dict['{}_start'.format(param)][0])
                end = mapping(params_dict['{}_end'.format(param)][0])
                increment = mapping(params_dict['{}_increment'.format(param)][0])

                parameters[param] = np.arange(start, end, increment).tolist()

            # do special parsing for MLPClassifier
            if self.classifier == 'MLPClassifier':
                hidden_layer_sizes = []

                for num_hidden_layers in parameters['num_hidden_layers']:
                    hidden_layer_sizes.extend(
                        list((i,)*num_hidden_layers for i in parameters['num_hidden_nodes']))

                del parameters['num_hidden_nodes']
                del parameters['num_hidden_layers']
                parameters['hidden_layer_sizes'] = hidden_layer_sizes
        else:
            mode = 'normal'
            parameters = {}

        log.debug('Classifier mode: %s', mode)
        log.debug('Parameters for \'%s\': %s', self.classifier, parameters)

        return mode, parameters

    def write_grid_search_results(self, best_params, save_to):
        """
        Write grid search results to file.

        Inputs:
            best_params: (dict) Dictionary where key is the parameter
                name and value is the actual parameter value to use.
            save_to: (str) Filepath to save the parameters to.
        """
        if best_params:
            best_params = {k.replace('SFS__estimator__{}__'.format(self.classifier), ''): v
                           for k, v in best_params.items()}

            section = '{}_Best'.format(self.classifier)
            self.configparser.append(section, best_params)

        self.configparser.write(save_to)

    def get_mode(self):
        """
        Get mode of classifier.

        Returns:
            (str) 'grid' | 'normal'
        """
        return self.mode

    def get_params(self):
        """
        Get parameters read from the config file.

        Returns:
            (dict) Dictionary where key is the parameter
                name and value is the actual parameter value to use.
        """
        return self.parameters

    def get_classifier(self):
        """
        Get classifier.

        Returns:
            Classifier object.
        """
        return self.clf

    def analyze_feature_selection(self, sfs):
        """
        Analyze sequential feature selection results.

        Inputs:
            sfs: (SequentialFeatureSelector) Fitted object.
        """
        metric_dict = sfs.get_metric_dict()

        pd_metric = pd.DataFrame.from_dict(metric_dict).T
        pd_metric = pd_metric.sort_index()

        log.debug('Feature selection metric: %s', pd_metric)
        log.info('Selected features: %s', sfs.k_feature_names_)

        rank = []
        feature_names = pd_metric['feature_names'].tolist()
        for i in range(len(feature_names)-1):
            if i == 0:
                rank.append(feature_names[i][0])

            difference = list(set(feature_names[i+1]) - set(feature_names[i]))
            rank.append(difference[0])

        log.info('Feature rank from high to low: %s', rank)

        if self.visualization_dir:
            plot_sfs(
                metric_dict,
                rank,
                title='Sequential Backward Selection (w. StdDev)',
                save_to=self.visualization_dir,
                filename='sfs_{}.png'.format(self.classifier))

        return rank

    def fit(self, X, y):
        """
        Fit classifier.

        Inputs:
            X: (array-like) Training data.
            y: (array-like) Target values.
        """
        self.clf.fit(X, y)

    def predict(self, X):
        """
        Do prediction.

        Inputs:
            X: (array-like) Training data.
        """
        return self.clf.predict(X)

    def predict_proba(self, X):
        """
        Predict probability.

        Inputs:
            X: (array-like) Training data.
        """
        return self.clf.predict_proba(X)

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
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# local imports
from utils.utilities import check_str_type


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

        # read which classifier we are using
        self.classifier = self.configparser.get_str('classifier')

        # parse mode (gridsearch or normal) and get parameters
        self.mode, self.parameters = self._parse_param()

        # assign appropriate classifier
        self.clf = self._build_pipeline()

    def _build_pipeline(self):
        # assign appropriate classifier
        if self.classifier.lower() == 'dummyclassifier':
            clf = DummyClassifier(strategy='most_frequent')
        elif self.classifier.lower() == 'decisiontreeclassifier':
            clf = DecisionTreeClassifier(
                **self.parameters if not self.mode == 'grid' else {})
        elif self.classifier.lower() == 'gaussiannb':
            clf = GaussianNB()
        elif self.classifier.lower() == 'multinomialnb':
            clf = MultinomialNB()
        elif self.classifier.lower() == 'categoricalnb':
            clf = CategoricalNB()
        elif self.classifier.lower() == 'svc':
            clf = SVC(probability=True, **self.parameters)
        elif self.classifier.lower() == 'adaboostclassifier':
            clf = AdaBoostClassifier(
                **self.parameters if not self.mode == 'grid' else {})
        elif self.classifier.lower() == 'randomforestclassifier':
            clf = RandomForestClassifier(
                n_jobs=-1, **self.parameters if not self.mode == 'grid' else {})
        elif self.classifier.lower() == 'mlpclassifier':
            clf = MLPClassifier(
                early_stopping=True, **self.parameters if not self.mode == 'grid' else {})
        else:
            raise ValueError('Invalid classifier: {}'.format(self.classifier))

        log.info('Selected classifier: %s', self.classifier)
        log.debug('Classifier info: %s', clf)

        if self.configparser.get_bool('SMOTE'):
            smote = SMOTE(sampling_strategy='minority')
            clf = Pipeline([('SMOTE', smote), (self.classifier, clf)])

        return clf

    def _parse_param(self):
        sections = self.configparser.sections()
        grid_search_section = '{}_GridSearch'.format(self.classifier)
        best_result_section = '{}_Best'.format(self.classifier)

        parameters = {}

        if best_result_section in sections:
            mode = 'normal'

            params_dict = self.configparser.get_section_as_dict(section=best_result_section)
            del params_dict['classifier']
            del params_dict['SMOTE']

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
            del params_dict['SMOTE']

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
        if best_params:
            if self.configparser.get_bool('SMOTE'):
                best_params = {k.replace('{}__'.format(self.classifier), ''): v
                               for k, v in best_params.items()}

            section = '{}_Best'.format(self.classifier)
            self.configparser.append(section, best_params)

        self.configparser.write(save_to)

    def get_mode(self):
        return self.mode

    def get_params(self):
        if self.configparser.get_bool('SMOTE'):
            return {'{}__{}'.format(self.classifier, k): v
                    for k, v in self.parameters.items()}
        else:
            return self.parameters

    def get_classifier(self):
        return self.clf

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def score(self, X, y):
        return self.clf.score(X, y)

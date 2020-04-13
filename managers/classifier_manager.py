"""
Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Classifier manager.

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
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# local imports
from utils.config_parser import ConfigParser
from utils.utilities import check_str_type


class ClassifierManager:
    """
    All about the classifiers.
    """

    def __init__(self, configfile):
        """
        Class initializer.

        Inputs:
            configfile: (str) Configuration file path.
        """
        # load config parser
        self.configfile = configfile
        self.configparser = ConfigParser(configfile)

        # read which classifier we are using
        self.classifier = self.configparser.get_str('classifier')

        # parse mode (gridsearch or normal) and get parameters
        self.mode, self.parameters = self._parse_param()

        # assign appropriate classifier
        if self.classifier.lower() == 'dummyclassifier':
            self.clf = DummyClassifier(strategy='most_frequent')
        elif self.classifier.lower() == 'decisiontreeclassifier':
            self.clf = DecisionTreeClassifier(
                **self.parameters if not self.mode == 'grid' else {})
        elif self.classifier.lower() == 'gaussiannb':
            self.clf = GaussianNB()
        elif self.classifier.lower() == 'multinomialnb':
            self.clf = MultinomialNB()
        elif self.classifier.lower() == 'categoricalnb':
            self.clf = CategoricalNB()
        elif self.classifier.lower() == 'svc':
            self.clf = SVC(**self.parameters)
        elif self.classifier.lower() == 'adaboostclassifier':
            self.clf = AdaBoostClassifier(
                **self.parameters if not self.mode == 'grid' else {})
        elif self.classifier.lower() == 'randomforestclassifier':
            self.clf = RandomForestClassifier(
                n_jobs=-1, **self.parameters if not self.mode == 'grid' else {})
        elif self.classifier.lower() == 'mlp':
            self.clf = MLPClassifier(**self.parameters if not self.mode == 'grid' else {})
            # self.clf = MLPClassifier(hidden_layer_sizes=(50, 50, 50), alpha=0.01, max_iter=1000)
        else:
            raise ValueError('Invalid classifier: {}'.format(self.classifier))

    def _parse_param(self):
        sections = self.configparser.sections()
        grid_search_section = '{}_GridSearch'.format(self.classifier)
        best_result_section = '{}_Best'.format(self.classifier)

        parameters = {}

        if best_result_section in sections:
            mode = 'normal'

            params_dict = self.configparser.get_section_as_dict(section=best_result_section)
            del params_dict['classifier']

            for key, value in params_dict.items():
                dtype = check_str_type(value[0])

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
        else:
            mode = 'normal'

        log.debug('Classifier mode: %s', mode)
        log.debug('Parameters for \'%s\': %s', self.classifier, parameters)

        return mode, parameters

    def write_grid_search_results(self, best_params, save_to):
        assert self.mode == 'grid'

        section = '{}_Best'.format(self.classifier)
        self.configparser.append(section, best_params)
        self.configparser.write(save_to)

    def get_mode(self):
        return self.mode

    def get_params(self):
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

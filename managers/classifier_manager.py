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
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# local imports
from utils.config_parser import ConfigParser


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
        configparser = ConfigParser(configfile)

        # read the configuration file
        self.classifier = configparser.get_str('classifier')

        if self.classifier.lower() == 'dummyclassifier':
            self.clf = DummyClassifier(strategy='most_frequent')
        elif self.classifier.lower() == 'decisiontree':
            self.clf = DecisionTreeClassifier()
        elif self.classifier.lower() == 'gaussiannb':
            self.clf = GaussianNB()
        elif self.classifier.lower() == 'multinomialnb':
            self.clf = MultinomialNB()
        elif self.classifier.lower() == 'bernoullinb':
            self.clf = BernoulliNB()
        elif self.classifier.lower() == 'categoricalnb':
            self.clf = CategoricalNB()
        elif self.classifier.lower() == 'svc':
            self.clf = SVC()
        elif self.classifier.lower() == 'adaboostclassifier':
            self.clf = AdaBoostClassifier()
        elif self.classifier.lower() == 'randomforestclassifier':
            self.clf = RandomForestClassifier()
        elif self.classifier.lower() == 'mlp':
            self.clf = MLPClassifier(hidden_layer_sizes=(50, 50, 50), alpha=0.01, max_iter=1000)
        else:
            raise ValueError('Invalid classifier: {}'.format(self.classifier))

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

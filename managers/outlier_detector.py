# standard imports
import logging as log

# third party imports
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor


def convert_index_2_bool(index):
    return [True if i == 1 else False for i in index]


def isolation_forest(pd_data):
    clf = IsolationForest().fit(pd_data)
    outliers = convert_index_2_bool(clf.predict(pd_data).tolist())

    log.debug('Number of outliers detected using Isolation Forest: %d', outliers.count(False))

    return outliers


def one_class_svm(pd_data):
    clf = OneClassSVM(gamma='auto').fit(pd_data)
    outliers = convert_index_2_bool(clf.predict(pd_data).tolist())

    log.debug('Number of outliers detected using One Class SVM: %d', outliers.count(False))

    return outliers


def local_outlier_factor(pd_data, n_neighbors=2):
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    outliers = convert_index_2_bool(clf.fit_predict(pd_data).tolist())

    log.debug('Number of outliers detected using Local Outlier Factor: %d', outliers.count(False))

    return outliers

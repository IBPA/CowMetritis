"""
Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Collection of wrapper functions for outlier detection.

To-do:
"""
# standard imports
import logging as log

# third party imports
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def convert_index_2_bool(index):
    """
    Convert integer style outlier / inlier index to boolean.

    Inputs:
        index: (list) -1 for outliers and 1 for inliers.

    Returns:
        (list) False for outliers and True for inliers.
    """
    return [True if i == 1 else False for i in index]


def isolation_forest(pd_data, random_state=None):
    """
    Detect outliers using the Isolation Forest algorithm.

    Inputs:
        pd_data: (DataFrame) Input data.
        random_state: (int, optional) Seed of the pseudo
            random number generator to use.

    Returns:
        (list) False for outliers and True for inliers.
    """
    clf = IsolationForest(n_jobs=-1, random_state=random_state).fit(pd_data)
    outliers = convert_index_2_bool(clf.predict(pd_data).tolist())

    log.debug('Number of outliers detected using Isolation Forest: %d', outliers.count(False))

    return outliers


def one_class_svm(pd_data):
    """
    Detect outliers using the One Class SVM.

    Inputs:
        pd_data: (DataFrame) Input data.

    Returns:
        (list) False for outliers and True for inliers.
    """
    clf = OneClassSVM(gamma='auto').fit(pd_data)
    outliers = convert_index_2_bool(clf.predict(pd_data).tolist())

    log.debug('Number of outliers detected using One Class SVM: %d', outliers.count(False))

    return outliers


def local_outlier_factor(pd_data):
    """
    Detect outliers using the LOF algorithm.

    Inputs:
        pd_data: (DataFrame) Input data.

    Returns:
        (list) False for outliers and True for inliers.
    """
    clf = LocalOutlierFactor(n_jobs=-1)
    outliers = convert_index_2_bool(clf.fit_predict(pd_data).tolist())

    log.debug('Number of outliers detected using Local Outlier Factor: %d', outliers.count(False))

    return outliers

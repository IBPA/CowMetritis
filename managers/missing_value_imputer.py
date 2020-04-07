"""
Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Collection of wrapper functions for missing value imputation.

To-do:
"""
# third party imports
from missingpy import MissForest
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer


def knn_imputer(pd_data):
    """
    Impute missing values using KNN.

    Inputs:
        pd_data: (DataFrame) Data containing missing values.

    Returns:
        pd_imputed: (DataFrame) Data with missing values imputed.
    """
    imputer = KNNImputer()

    pd_imputed = pd.DataFrame(
        imputer.fit_transform(pd_data),
        index=pd_data.index,
        columns=pd_data.columns)

    return pd_imputed


def iterative_imputer(pd_data):
    """
    Impute missing values using the multivariate imputer
    that estimates each feature from all the others.

    Inputs:
        pd_data: (DataFrame) Data containing missing values.

    Returns:
        pd_imputed: (DataFrame) Data with missing values imputed.
    """
    imputer = IterativeImputer()

    pd_imputed = pd.DataFrame(
        imputer.fit_transform(pd_data),
        index=pd_data.index,
        columns=pd_data.columns)

    return pd_imputed


def missforest_imputer(pd_data):
    """
    Impute missing values using the MissForest imputer.

    Inputs:
        pd_data: (DataFrame) Data containing missing values.

    Returns:
        pd_imputed: (DataFrame) Data with missing values imputed.
    """
    imputer = MissForest()

    pd_imputed = pd.DataFrame(
        imputer.fit_transform(pd_data),
        index=pd_data.index,
        columns=pd_data.columns)

    return pd_imputed

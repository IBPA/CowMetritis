"""
Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:

To-do:
"""
# third party imports
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer


def knn_imputer(pd_data, n_neighbors=2):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    pd_imputed = pd.DataFrame(
        imputer.fit_transform(pd_data),
        index=pd_data.index,
        columns=pd_data.columns)

    return pd_imputed


def iterative_imputer(pd_data, max_iter=10):
    imputer = IterativeImputer(max_iter=max_iter)
    pd_imputed = pd.DataFrame(
        imputer.fit_transform(pd_data),
        index=pd_data.index,
        columns=pd_data.columns)

    return pd_imputed

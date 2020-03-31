# standard imports
import logging as log

# third party imports
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor


def isolation_forest(pd_data):
	clf = IsolationForest().fit(pd_data)
	outliers = clf.predict(pd_data)

	log.debug('Number of outliers detected using Isolation Forest: %d',
		outliers.tolist().count(-1))

	return outliers

def one_class_svm(pd_data):
	clf = OneClassSVM(gamma='auto').fit(pd_data)
	outliers = clf.predict(pd_data)

	log.debug('Number of outliers detected using One Class SVM: %d',
		outliers.tolist().count(-1))

	return outliers

def local_outlier_factor(pd_data, n_neighbors=2):
	clf = LocalOutlierFactor(n_neighbors=n_neighbors)
	outliers = clf.fit_predict(pd_data)

	log.debug('Number of outliers detected using Local Outlier Factor: %d',
		outliers.tolist().count(-1))

	return outliers

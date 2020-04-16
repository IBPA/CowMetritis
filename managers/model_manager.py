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
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# local imports
from managers.preprocess_manager import PreprocessManager
from managers.classifier_manager import ClassifierManager
from utils.visualization import visualize_missing_values, plot_projection, plot_scatter_matrix
from utils.utilities import get_results_of_search


class ModelManager:
    """
    All about the classifiers.
    """

    def __init__(self):
        """
        Class initializer.

        Inputs:
        """
        pass

    def preprocess(self, preprocess_config, section='DEFAULT'):
        # init object for preprocessing
        pmanager = PreprocessManager(preprocess_config, section=section)

        # do we do visualization?
        visualization_dir = preprocess_config.get_str('visualization_dir')

        # read data and get independent / dependent variables
        pd_data = pmanager.read_data()
        X, y = pmanager.get_X_and_y(pd_data)

        # visualize missing values
        if visualization_dir:
            visualize_missing_values(X, visualization_dir)

        # scale features
        X = pmanager.scale_features(X)

        # impute missing value
        X = pmanager.impute_missing_values(X)

        # detect outliers
        outlier_index = pmanager.detect_outlier(X)

        # perform & visualize dimensionality reduction
        X_dr, reduction_mode = pmanager.reduce_dimension(X)

        if visualization_dir:
            plot_projection(X_dr, y, reduction_mode, visualization_dir, outlier_index)

        # remove outliers
        X, y = pmanager.remove_outlier(X, y, outlier_index)

        # plot scatter matrix of the data
        if visualization_dir:
            plot_scatter_matrix(X, y, visualization_dir)

        # do feature selection
        if visualization_dir:
            X = pmanager.feature_selection(X, y, visualization_dir)
        else:
            X = pmanager.feature_selection(X, y)

        return X, y

    def grid_search(self, X, y, scoring, classifier_config, updated_classifier_config):
        cmanager = ClassifierManager(classifier_config)

        # perform grid searching if specified in the config file
        if cmanager.get_mode() == 'grid':
            grid_search_cv = GridSearchCV(
                cmanager.get_classifier(),
                param_grid=cmanager.get_params(),
                scoring=scoring,
                n_jobs=-1,
                verbose=1)

            grid_search_cv.fit(X, y)
            best_params, best_score = get_results_of_search(
                grid_search_cv.cv_results_,
                report_score_using=scoring,
                scoring=scoring)
        else:
            log.info('Not doing any grid search for \'%s\'', cmanager.classifier)

            best_score, _, _ = self.run_model_cv(X, y, scoring, classifier_config)
            best_params = None

        log.info('Best %s score: %f', scoring, best_score)
        log.info('Best parameters: %s', best_params)

        # write grid search results to the config file
        cmanager.write_grid_search_results(
            best_params,
            updated_classifier_config)

        return best_score

    def run_model_cv(self, X, y, scoring, classifier_config):
        cmanager = ClassifierManager(classifier_config)

        if cmanager.get_mode() == 'grid':
            raise RuntimeError('Cannot run classifier \'{}\' without grid search results!'.format(
                classifier_config.get_str('classifier')))

        # run best model again to get metrics
        skf = StratifiedKFold(shuffle=True, random_state=1)

        scores = []
        y_trues = ()
        y_preds = ()
        y_probs = ()

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            cmanager.fit(X_train, y_train)
            y_pred = cmanager.predict(X_test)
            y_prob = pd.DataFrame(
                cmanager.predict_proba(X_test),
                columns=cmanager.get_classifier().classes_)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            log.debug('Confusion matrix (tp, fp, fn, tn): (%d, %d, %d, %d)', tp, fp, fn, tn)

            y_trues += (y_test,)
            y_preds += (y_pred,)
            y_probs += (y_prob,)

            if scoring.lower() == 'f1':
                scores.append(f1_score(y_test, y_pred))
            elif scoring.lower() == 'accuracy':
                scores.append(accuracy_score(y_test, y_pred))
            else:
                raise ValueError('Invalid scoring: {}'.format(scoring))

        log.info('%s for each fold: %s', scoring, scores)

        score_avg = np.mean(scores)
        score_std = np.std(scores)

        log.info('%s score: %f ± %f', scoring, score_avg, score_std)

        ys = (y_trues, y_preds, y_probs)

        return score_avg, score_std, ys

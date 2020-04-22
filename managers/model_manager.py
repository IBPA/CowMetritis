"""
Authors:
    Jason Youn - jyoun@ucdavis.edu

Description:
    Manager to do preprocessing and classification.

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
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# local imports
from managers.preprocess_manager import PreprocessManager
from managers.classifier_manager import ClassifierManager
from utils.visualization import visualize_missing_values, plot_projection, plot_scatter_matrix, plot_bic_fs
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

    def preprocess(self, preprocess_config, section='DEFAULT', final_model=False):
        """
        Run preprocessing (feature scaling -> missing value imputation -> outlie detection).

        Inputs:
            preprocess_config: (ConfigParser) Object containing configuration.
            section: (str, optional) If specified, run preprocessing using the
                classifier specific preprocessing combination.
            final_model: (bool, optional) If True, run preprocessing for final model.

        Returns:
            X: (DataFrame) Features.
            y: (DataFrame) Dependent variable.
        """
        # init object for preprocessing
        pmanager = PreprocessManager(preprocess_config, classifier=section)

        # do we do visualization?
        visualization_dir = preprocess_config.get_str('visualization_dir')

        # read data and get independent / dependent variables
        pd_data = pmanager.read_data()
        if final_model:
            X, y = pmanager.get_X(pd_data), None
        else:
            X, y = pmanager.get_X_and_y(pd_data)

            # visualize missing values
            visualize_missing_values(X, visualization_dir)

        # scale features
        X = pmanager.scale_features(X, final_model=final_model)

        if not final_model:
            # impute missing value
            X = pmanager.impute_missing_values(X)

            # detect outliers
            outlier_index = pmanager.detect_outlier(X)

            # perform & visualize dimensionality reduction
            X_dr, reduction_mode = pmanager.reduce_dimension(X)
            plot_projection(X_dr, y, reduction_mode, visualization_dir, outlier_index)

            # remove outliers
            X, y = pmanager.remove_outlier(X, y, outlier_index)

            # plot scatter matrix of the data
            plot_scatter_matrix(X, y, visualization_dir)

            # do feature analysis
            pmanager.feature_analysis(X, y, visualization_dir)

        return X, y

    def feature_selector(self, X, y, classifier_config, scoring='f1'):
        """
        Do recursive feature selection.

        Inputs:
            X: (DataFrame) Features.
            y: (DataFrame) Dependent variable.
            classifier_config: (ConfigParser) Object containing
                clasifier specific configuration.
            scoring: (str) 'f1' | 'accuracy'
        """
        cmanager = ClassifierManager(classifier_config)

        sfs = SFS(estimator=cmanager.get_classifier(),
                  k_features='parsimonious',
                  forward=False,
                  floating=False,
                  scoring=scoring,
                  cv=5,
                  n_jobs=-1,
                  verbose=1)

        sfs.fit(X, y)

        cmanager.analyze_feature_selection(sfs)
        X_selected = X[list(sfs.k_feature_names_)]

        return X_selected

    def grid_search(self, X, y, scoring, classifier_config, updated_classifier_config):
        """
        Do grid search if necessary and save the best parameters.

        Inputs:
            X: (DataFrame) Features.
            y: (DataFrame) Dependent variable.
            scoring: (str) 'f1' | 'accuracy'
            classifier_config: (ConfigParser) Object containing
                clasifier specific configuration.
            updated_classifier_config: (str) Filepath to save the
                best parameters found by grid search to.
        """
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

    def run_model_cv(self, X, y, scoring, classifier_config, random_state=1):
        """
        Run k-fold CV and report the results.

        Inputs:
            X: (DataFrame) Features.
            y: (DataFrame) Dependent variable.
            scoring: (str) 'f1' | 'accuracy'
            classifier_config: (ConfigParser) Object containing
                clasifier specific configuration.
            random_state: (int) Control reproducibility.
        """
        cmanager = ClassifierManager(classifier_config)

        if cmanager.get_mode() == 'grid':
            raise RuntimeError('Cannot run classifier \'{}\' without grid search results!'.format(
                classifier_config.get_str('classifier')))

        # run best model again to get metrics
        skf = StratifiedKFold(shuffle=True, random_state=random_state)

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

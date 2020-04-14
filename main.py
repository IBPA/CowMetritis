"""
Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:
    Main python file to run.

To-do:
"""
# standard imports
import argparse
import itertools
import logging as log

# third party imports
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

# local imports
from managers.preprocess_manager import PreprocessManager
from managers.classifier_manager import ClassifierManager
from utils.config_parser import ConfigParser
from utils.set_logging import set_logging
from utils.visualization import visualize_missing_values, plot_projection, plot_scatter_matrix
from utils.utilities import get_results_of_search

# global variables
DEFAULT_CONFIG_FILE = './config/main.ini'


def parse_argument():
    """
    Parse input arguments.

    Returns:
        - parsed arguments
    """
    parser = argparse.ArgumentParser(description='Description goes here.')

    parser.add_argument(
        '--config_file',
        default=DEFAULT_CONFIG_FILE,
        help='Path to the .ini configuration file.')

    return parser.parse_args()


def preprocess(main_config, preprocess_config):
    # do we do visualization?
    visualization = main_config.get_bool('visualization')

    # init object for preprocessing
    pmanager = PreprocessManager(preprocess_config)

    # read data and get independent / dependent variables
    pd_data = pmanager.read_data()
    X, y = pmanager.get_X_and_y(pd_data)

    # visualize missing values
    if visualization:
        visualize_missing_values(X, main_config.get_str('visualization_dir'))

    # scale features
    X = pmanager.scale_features(X)

    # impute missing value
    X = pmanager.impute_missing_values(X)

    # detect outliers
    outlier_index = pmanager.detect_outlier(X)

    # perform & visualize dimensionality reduction
    X_dr, reduction_mode = pmanager.reduce_dimension(X)

    if visualization:
        plot_projection(
            X_dr, y,
            reduction_mode,
            main_config.get_str('visualization_dir'),
            outlier_index)

    # remove outliers
    X, y = pmanager.remove_outlier(X, y, outlier_index)

    # plot scatter matrix of the data
    if visualization:
        plot_scatter_matrix(X, y, main_config.get_str('visualization_dir'))

    # do feature selection
    if visualization:
        X = pmanager.feature_selection(X, y, main_config.get_str('visualization_dir'))
    else:
        X = pmanager.feature_selection(X, y)

    return X, y


def run_model(X, y, main_config, classifier_config):
    cmanager = ClassifierManager(classifier_config)

    # perform grid searching if specified in the config file
    if cmanager.get_mode() == 'grid':
        grid_search = GridSearchCV(
            cmanager.get_classifier(),
            param_grid=cmanager.get_params(),
            scoring='f1',
            n_jobs=-1,
            verbose=1)

        grid_search.fit(X, y)
        best_params = get_results_of_search(grid_search.cv_results_, scoring='f1')

        # write grid search results to the config file
        cmanager.write_grid_search_results(
            best_params,
            main_config.get_str('updated_classifier_config'))

        # update the object with the new classifier using best parameters
        updated_classifier_config = ConfigParser(main_config.get_str('updated_classifier_config'))
        cmanager = ClassifierManager(updated_classifier_config)

    # run best model again to get metrics
    skf = StratifiedKFold(shuffle=True, random_state=0)

    X = X.to_numpy()
    y = y.to_numpy()

    f1_avg = 0
    accuracy_avg = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        cmanager.fit(X_train, y_train)
        y_pred = cmanager.predict(X_test)

        f1_avg += f1_score(y_test, y_pred)
        accuracy_avg += accuracy_score(y_test, y_pred)

    f1_avg /= 5
    accuracy_avg /= 5

    return f1_avg, accuracy_avg


def main():
    """
    Main function.
    """
    # parse args
    args = parse_argument()

    # load config files
    main_config = ConfigParser(args.config_file)
    preprocess_config = ConfigParser(main_config.get_str('preprocess_config'))
    classifier_config = ConfigParser(main_config.get_str('classifier_config'))

    # set logging
    set_logging(log_file=main_config.get_str('log_file'))

    # run models for all possible combination of preprocessing
    scale_modes = main_config.get_str_list('scale_mode')
    mvi_modes = main_config.get_str_list('mvi_mode')
    outlier_modes = main_config.get_str_list('outlier_mode')
    classifiers = main_config.get_str_list('classifier')

    classifier_f1_dict = {classifier: 0 for classifier in classifiers}
    classifier_accuracy_dict = {classifier: 0 for classifier in classifiers}
    classifier_best_preprocessing_f1_dict = {classifier: None for classifier in classifiers}
    classifier_best_preprocessing_accuracy_dict = {classifier: None for classifier in classifiers}
    all_combinations = [scale_modes, mvi_modes, outlier_modes, classifiers]

    for combination in list(itertools.product(*all_combinations)):
        # unpack the tuple
        scale_mode = combination[0]
        mvi_mode = combination[1]
        outlier_mode = combination[2]
        classifier = combination[3]

        combination_str_joined = ', '.join(list(combination))
        log.info('Running grid search: (%s)', combination_str_joined)

        if classifier in ['MultinomialNB', 'CategoricalNB'] and scale_mode != 'minmax':
            log.info('Skipping this combination...')

        preprocess_config.overwrite('scale_mode', scale_mode)
        preprocess_config.overwrite('mvi_mode', mvi_mode)
        preprocess_config.overwrite('outlier_mode', outlier_mode)
        classifier_config.overwrite('classifier', classifier)

        # perform preprocessing
        X, y = preprocess(main_config, preprocess_config)

        # run classification model
        f1, accuracy = run_model(X, y, main_config, classifier_config)
        log.info('F1: %f', f1)
        log.info('Accuracy: %f', accuracy)

        # update the best preprocessing combination
        if classifier_f1_dict[classifier] < f1:
            classifier_f1_dict[classifier] = f1
            classifier_best_preprocessing_f1_dict[classifier] = combination_str_joined

        if classifier_accuracy_dict[classifier] < accuracy:
            classifier_accuracy_dict[classifier] = accuracy
            classifier_best_preprocessing_accuracy_dict[classifier] = combination_str_joined

    log.info('Best F1 score for each classifier: %s',
             classifier_f1_dict)
    log.info('Best accuracy score for each classifier: %s',
             classifier_accuracy_dict)
    log.info('Preprocessing combination of the best F1 score for each classifier: %s',
             classifier_best_preprocessing_f1_dict)
    log.info('Preprocessing combination of the best accuracy score for each classifier: %s',
             classifier_best_preprocessing_accuracy_dict)


if __name__ == '__main__':
    main()

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
    # init object for preprocessing
    pmanager = PreprocessManager(preprocess_config)

    # read data and get independent / dependent variables
    pd_data = pmanager.read_data()
    X, y = pmanager.get_X_and_y(pd_data)

    # visualize missing values
    visualize_missing_values(X, main_config.get_str('visualization_dir'))

    # scale features
    X = pmanager.scale_features(X)

    # impute missing value
    X = pmanager.impute_missing_values(X)

    # detect outliers
    outlier_index = pmanager.detect_outlier(X)

    # perform & visualize dimensionality reduction
    X_dr, reduction_mode = pmanager.reduce_dimension(X)
    plot_projection(
        X_dr, y,
        reduction_mode,
        main_config.get_str('visualization_dir'),
        outlier_index)

    # remove outliers
    X, y = pmanager.remove_outlier(X, y, outlier_index)

    # plot scatter matrix of the data
    plot_scatter_matrix(X, y, main_config.get_str('visualization_dir'))

    # do feature selection
    X = pmanager.feature_selection(X, y, main_config.get_str('visualization_dir'))

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

        # updated the object with the new classifier using best parameters
        cmanager = ClassifierManager(main_config.get_str('updated_classifier_config'))

    skf = StratifiedKFold(shuffle=True)

    X = X.to_numpy()
    y = y.to_numpy()

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        cmanager.fit(X_train, y_train)
        y_pred = cmanager.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f1, accuracy)


def main():
    """
    Main function.
    """
    # set log, parse args, and read configuration
    set_logging()
    args = parse_argument()

    # load config files
    main_config = ConfigParser(args.config_file)
    preprocess_config = ConfigParser(main_config.get_str('preprocess_config'))
    classifier_config = ConfigParser(main_config.get_str('classifier_config'))

    # run models for all possible combination of preprocessing
    scale_modes = main_config.get_str_list('scale_mode')
    mvi_modes = main_config.get_str_list('mvi_mode')
    outlier_modes = main_config.get_str_list('outlier_mode')
    classifiers = main_config.get_str_list('classifier')

    all_combinations = [scale_modes, mvi_modes, outlier_modes, classifiers]

    for scale_mode, mvi_mode, outlier_mode, classifier in list(itertools.product(*all_combinations)):
        log.info('Running grid search: (%s, %s, %s, %s)',
                 scale_mode, mvi_mode, outlier_mode, classifier)

        if classifier in ['MultinomialNB', 'CategoricalNB'] and scale_mode != 'minmax':
            continue

        preprocess_config.overwrite('scale_mode', scale_mode)
        preprocess_config.overwrite('mvi_mode', mvi_mode)
        preprocess_config.overwrite('outlier_mode', outlier_mode)
        classifier_config.overwrite('classifier', classifier)

        # perform preprocessing
        X, y = preprocess(main_config, preprocess_config)

        # run classification model
        run_model(X, y, main_config, classifier_config)


if __name__ == '__main__':
    main()

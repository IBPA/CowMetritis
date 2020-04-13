"""
Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:
    Main python file to run.

To-do:
"""
# standard imports
import argparse

# third party imports
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

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


def preprocess(configparser):
    # init object for preprocessing
    pmanager = PreprocessManager(configparser.get_str('preprocess_config'))

    # read data and get independent / dependent variables
    pd_data = pmanager.read_data()
    X, y = pmanager.get_X_and_y(pd_data)

    # visualize missing values
    visualize_missing_values(X, configparser.get_str('visualization_dir'))

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
        configparser.get_str('visualization_dir'),
        outlier_index)

    # remove outliers
    X, y = pmanager.remove_outlier(X, y, outlier_index)

    # plot scatter matrix of the data
    plot_scatter_matrix(X, y, configparser.get_str('visualization_dir'))

    # do feature selection
    X = pmanager.feature_selection(X, y, configparser.get_str('visualization_dir'))

    return X, y


def run_model(X, y, configparser):
    cmanager = ClassifierManager(configparser.get_str('classifier_config'))

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
            configparser.get_str('updated_classifier_config'))

        # updated the object with the new classifier using best parameters
        cmanager = ClassifierManager(configparser.get_str('updated_classifier_config'))

    # skf = StratifiedKFold(shuffle=True)

    # X = X.to_numpy()
    # y = y.to_numpy()

    # accuracy = 0

    # for train_index, test_index in skf.split(X, y):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]

    #     cmanager.fit(X_train, y_train)

    #     accuracy += cmanager.score(X_test, y_test)

    # print(accuracy / 5)


def main():
    """
    Main function.
    """
    # set log, parse args, and read configuration
    set_logging()
    args = parse_argument()
    configparser = ConfigParser(args.config_file)

    # perform preprocessing
    X, y = preprocess(configparser)

    # run classification model
    run_model(X, y, configparser)


if __name__ == '__main__':
    main()

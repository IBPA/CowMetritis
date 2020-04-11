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
from sklearn.model_selection import StratifiedKFold
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

# local imports
from managers.preprocess_manager import PreprocessManager
from managers.classifier_manager import ClassifierManager
from utils.config_parser import ConfigParser
from utils.set_logging import set_logging
from utils.visualization import visualize_missing_values, plot_projection, plot_scatter_matrix

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

    return X, y


def run_model(X, y, configparser):
    cmanager = ClassifierManager(configparser.get_str('classifier_config'))



    # from sklearn.model_selection import StratifiedKFold
    # skf = StratifiedKFold(shuffle=True)

    # X = X.to_numpy()
    # y = y.to_numpy()

    # for train_index, test_index in skf.split(X, y):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]

    #     cmanager.fit(X_train, y_train)

    #     print(cmanager.score(X_test, y_test))


    selector = RFECV(
        cmanager.get_classifier(),
        step=1,
        min_features_to_select=4,
        cv=5,
        scoring='accuracy',
        n_jobs=-1)
    selector = selector.fit(X, y)
    print(selector.support_)
    print(selector.ranking_)

    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
    plt.show()




    # efs = EFS(cmanager.get_classifier(),
    #           min_features=1,
    #           max_features=X.shape[1],
    #           scoring='accuracy',
    #           print_progress=True,
    #           cv=5,
    #           n_jobs=-1)

    # efs = efs.fit(X, y)

    # print('Best accuracy score: %.2f' % efs.best_score_)
    # print('Best subset (indices):', efs.best_idx_)
    # print('Best subset (corresponding names):', efs.best_feature_names_)


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

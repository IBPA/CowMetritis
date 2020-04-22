"""
Authors:
    Jason Youn -jyoun@ucdavis.edu

Description:
    Main python file to run.

To-do:
"""
# standard imports
import argparse
import logging as log
import os

# third party imports
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import ttest_rel

# local imports
from managers.classifier_manager import ClassifierManager
from managers.model_manager import ModelManager
from utils.config_parser import ConfigParser
from utils.set_logging import set_logging
from utils.utilities import create_dir, dir_exists, save_pkl, load_pkl

# global variables
DEFAULT_CONFIG_FILE = './config/recommender.ini'


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


def save_models(classifier,
                pre_built_models_dir,
                main_config,
                model_manager,
                num_classifiers):
    log.info('Pre-built model directory specified for %s does not exist.', classifier)
    log.info('Building models again.')

    # create directory
    create_dir(pre_built_models_dir)

    # load config parsers
    preprocess_config = ConfigParser(main_config.get_str('preprocess_config'))
    classifier_config = ConfigParser(main_config.get_str('classifier_config'))
    classifier_config.overwrite('classifier', classifier)

    # perform preprocessing
    X, y = model_manager.preprocess(preprocess_config, section=classifier)

    # select subset of features if requested
    selected_features = main_config.get_str_list('selected_features')
    if selected_features:
        log.info('Selecting subset of features: %s', selected_features)
        X = X[selected_features]

    # train multiple classifiers
    for i in range(num_classifiers):
        log.debug('Processing classifier %d/%s', i+1, num_classifiers)

        cmanager = ClassifierManager(classifier_config)
        clf = CalibratedClassifierCV(cmanager.get_classifier(), method='sigmoid', cv=5)
        clf.fit(X, y)

        save_pkl(
            clf,
            os.path.join(pre_built_models_dir, 'model_{}.pkl'.format(i)))


def make_recommendation(classifier,
                        pre_built_models_dir,
                        main_config,
                        model_manager,
                        num_classifiers):
    if not dir_exists(pre_built_models_dir):
        raise RuntimeError('Pre-built model directory does not exist!')

    log.info('Using pre-built model directory: %s', pre_built_models_dir)

    # load config parsers
    preprocess_config = ConfigParser(main_config.get_str('preprocess_recommender_config'))
    classifier_config = ConfigParser(main_config.get_str('classifier_config'))
    classifier_config.overwrite('classifier', classifier)

    # perform preprocessing
    X, y = model_manager.preprocess(preprocess_config, section=classifier, final_model=True)

    # select subset of features if requested
    selected_features = main_config.get_str_list('selected_features')
    if selected_features:
        log.info('Selecting subset of features: %s', selected_features)
        X = X[selected_features]

    def _revert_column(pd_data):
        values = list(set(pd_data.tolist()))

        replace_dict = {}
        for value in values:
            replace_dict[value] = list(filter(lambda a: a != value, values))[0]

        return pd_data.replace(to_replace=replace_dict)

    # get test data and its inverse for TRT column
    X_inv = X.copy()
    X_inv['TRT'] = _revert_column(X_inv['TRT'])
    pos_trt_idx = (X['TRT'] == 1.0)

    y_probs = []
    y_probs_inv = []
    for i in range(num_classifiers):
        log.debug('Processing classifier %d/%s', i+1, num_classifiers)

        classifier_filepath = os.path.join(pre_built_models_dir, 'model_{}.pkl'.format(i))
        log.debug('Loading classifier: %s', classifier_filepath)
        clf = load_pkl(classifier_filepath)

        y_probs.append(clf.predict_proba(X)[:, 1])
        y_probs_inv.append(clf.predict_proba(X_inv)[:, 1])

    y_probs = pd.DataFrame(y_probs).T
    y_probs.index = X.index
    y_probs_inv = pd.DataFrame(y_probs_inv).T
    y_probs_inv.index = X.index

    # make recommendation
    y_probs_avg = y_probs.mean(axis=1)
    y_probs_inv_avg = y_probs_inv.mean(axis=1)

    y_probs_avg_diff = y_probs_avg - y_probs_inv_avg
    inv_minus_pos = y_probs_inv_avg - y_probs_avg
    y_probs_avg_diff[~pos_trt_idx] = inv_minus_pos[~pos_trt_idx]

    pval = pd.Series(index=X.index)
    for index, _ in pval.items():
        _, pval[index] = ttest_rel(y_probs.loc[index], y_probs_inv.loc[index])

    # calculate y_probs_trt / right now it's y_probs  ################
    pd_concat = pd.concat(
        [pos_trt_idx, y_probs_avg, y_probs_inv_avg, y_probs_avg_diff, pval], axis=1)
    pd_concat.columns = ['pos_trt', 'y_probs_avg', 'y_probs_inv_avg', 'y_probs_avg_diff', 'pval']

    print(pd_concat)


def main():
    """
    Main function.
    """
    # parse args
    args = parse_argument()

    # load main config file and set logging
    main_config = ConfigParser(args.config_file)
    set_logging(log_file=main_config.get_str('log_file'))

    # initialize model manager object
    model_manager = ModelManager()

    # parse config
    classifier = main_config.get_str('classifier')
    pre_built_models_dir = os.path.join(main_config.get_str('pre_built_models_dir'), classifier)
    num_classifiers = main_config.get_int('num_classifiers')

    # we need to build the models first if they do not exist
    if not dir_exists(pre_built_models_dir):
        save_models(
            classifier,
            pre_built_models_dir,
            main_config,
            model_manager,
            num_classifiers)

    make_recommendation(
        classifier,
        pre_built_models_dir,
        main_config,
        model_manager,
        num_classifiers)


if __name__ == '__main__':
    main()

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
import sys

# third party imports
import matplotlib.pyplot as plt

# local imports
from managers.model_manager import ModelManager
from utils.config_parser import ConfigParser
from utils.set_logging import set_logging
from utils.visualization import plot_pr, plot_roc, save_figure

# global variables
DEFAULT_CONFIG_FILE = './config/analysis.ini'


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

    # perform analysis on these classifiers
    classifiers = main_config.get_str_list('classifier')

    # do prediction
    classifiers_ys = {}
    for classifier in classifiers:
        log.info('Running model for classifier \'%s\'', classifier)

        # load config parsers
        preprocess_config = ConfigParser(main_config.get_str('preprocess_config'))
        classifier_config = ConfigParser(main_config.get_str('classifier_config'))

        # perform preprocessing
        X, y = model_manager.preprocess(preprocess_config, section=classifier)

        # run classification model
        classifier_config.overwrite('classifier', classifier)

        X = model_manager.feature_selector(X, y, classifier_config)

        score_avg, score_std, ys = model_manager.run_model_cv(X, y, 'f1', classifier_config)

        classifiers_ys[classifier] = ys

    # plot PR curve
    fig = plt.figure()

    lines = []
    labels = []
    for classifier, ys in classifiers_ys.items():
        y_trues, y_preds, y_probs = ys

        y_probs_1 = tuple(y_prob[1].to_numpy() for y_prob in y_probs)
        line, label = plot_pr(y_trues, y_probs_1, classifier)

        lines.append(line)
        labels.append(label)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend(lines, labels, loc='lower right', prop={'size': 8})

    save_figure(fig, os.path.join(main_config.get_str('visualization_dir'), 'pr_curve.png'))

    # plot ROC curve
    fig = plt.figure()

    lines = []
    labels = []
    for classifier, ys in classifiers_ys.items():
        y_trues, y_preds, y_probs = ys

        y_probs_1 = tuple(y_prob[1].to_numpy() for y_prob in y_probs)
        line, label = plot_roc(y_trues, y_probs_1, classifier)

        lines.append(line)
        labels.append(label)

    # plt.plot([0, 1], [0, 1], color='k', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(lines, labels, loc='lower right', prop={'size': 8})

    save_figure(fig, os.path.join(main_config.get_str('visualization_dir'), 'roc_curve.png'))


if __name__ == '__main__':
    main()

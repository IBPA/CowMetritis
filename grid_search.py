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

# local imports
from managers.model_manager import ModelManager
from utils.config_parser import ConfigParser
from utils.set_logging import set_logging

# global variables
DEFAULT_CONFIG_FILE = './config/grid_search.ini'


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

    # run models for all possible combination of preprocessing
    scale_modes = main_config.get_str_list('scale_mode')
    mvi_modes = main_config.get_str_list('mvi_mode')
    outlier_modes = main_config.get_str_list('outlier_mode')
    classifiers = main_config.get_str_list('classifier')

    classifier_score_dict = {classifier: 0 for classifier in classifiers}
    classifier_best_combination_dict = {classifier: None for classifier in classifiers}
    all_combinations = [scale_modes, mvi_modes, outlier_modes, classifiers]
    all_combinations = list(itertools.product(*all_combinations))
    failed_combinations = []

    for idx, combination in enumerate(all_combinations):
        # unpack the tuple
        scale_mode = combination[0]
        mvi_mode = combination[1]
        outlier_mode = combination[2]
        classifier = combination[3]

        # log current combination
        combination_str_joined = ', '.join(list(combination))
        log.info('Running grid search %d/%d: (%s)',
                 idx+1, len(all_combinations), combination_str_joined)

        # some classifiers must use minmax scaler
        if classifier in ['MultinomialNB', 'CategoricalNB'] and scale_mode != 'minmax':
            log.info('Skipping this combination...')
            continue

        # overwrite the config file using the current combination
        preprocess_config = ConfigParser(main_config.get_str('preprocess_config'))
        classifier_config = ConfigParser(main_config.get_str('classifier_config'))

        preprocess_config.overwrite('scale_mode', scale_mode)
        preprocess_config.overwrite('mvi_mode', mvi_mode)
        preprocess_config.overwrite('outlier_mode', outlier_mode)
        classifier_config.overwrite('classifier', classifier)

        # perform preprocessing
        X, y = model_manager.preprocess(preprocess_config)

        # run classification model
        try:
            score = model_manager.grid_search(
                X, y,
                main_config.get_str('optimize_scoring'),
                classifier_config,
                main_config.get_str('updated_classifier_config'))
        except (IndexError, ValueError) as e:
            failed_combinations.append(combination_str_joined)
            log.error(e)
            continue

        # update the best preprocessing combination
        if classifier_score_dict[classifier] < score:
            classifier_score_dict[classifier] = score
            classifier_best_combination_dict[classifier] = combination_str_joined

    log.info('Best %s score for each classifier: %s',
             main_config.get_str('optimize_scoring'),
             classifier_score_dict)

    log.info('Preprocessing combination of the best %s score for each classifier: %s',
             main_config.get_str('optimize_scoring'),
             classifier_best_combination_dict)

    log.info('%d failed combinations: %s', len(failed_combinations), failed_combinations)


if __name__ == '__main__':
    main()

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

# third party imports
import matplotlib.pyplot as plt
import numpy as np

# local imports
from managers.model_manager import ModelManager
from utils.config_parser import ConfigParser
from utils.set_logging import set_logging
from utils.visualization import plot_pr, save_figure
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score

# global variables
DEFAULT_CONFIG_FILE = './config/best_model.ini'


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


def plot_pr_print_cm(baseline_classifier, best_classifier, main_config, model_manager):
    classifiers_ys = {}

    for classifier in [baseline_classifier, best_classifier]:
        log.info('Running model for classifier \'%s\'', classifier)

        # load config parsers
        preprocess_config = ConfigParser(main_config.get_str('preprocess_config'))
        classifier_config = ConfigParser(main_config.get_str('classifier_config'))

        # perform preprocessing
        X, y = model_manager.preprocess(preprocess_config, section=classifier)

        # select subset of features if requested
        selected_features = main_config.get_str_list('selected_features')
        if selected_features:
            log.info('Selecting subset of features: %s', selected_features)
            X = X[selected_features]

        # run classification model
        classifier_config.overwrite('classifier', classifier)

        score_avg, score_std, ys = model_manager.run_model_cv(
            X, y, 'f1', classifier_config)

        classifiers_ys[classifier] = ys

    # confusion matrix
    (y_trues, y_preds, y_probs) = classifiers_ys[best_classifier]

    tn = []
    fp = []
    fn = []
    tp = []

    pred_pos = []
    pred_neg = []
    known_pos = []
    known_neg = []

    f1 = []
    precision = []
    recall = []
    specificity = []
    npv = []
    fdr = []
    accuracy = []
    for fold in range(len(y_trues)):
        cm_result = confusion_matrix(y_trues[fold], y_preds[fold]).ravel()
        tn.append(cm_result[0])
        fp.append(cm_result[1])
        fn.append(cm_result[2])
        tp.append(cm_result[3])

        pred_pos.append(cm_result[3] + cm_result[1])
        pred_neg.append(cm_result[2] + cm_result[0])
        known_pos.append(cm_result[3] + cm_result[2])
        known_neg.append(cm_result[1] + cm_result[0])

        f1.append(f1_score(y_trues[fold], y_preds[fold]))
        precision.append(precision_score(y_trues[fold], y_preds[fold], average='binary'))
        recall.append(recall_score(y_trues[fold], y_preds[fold], average='binary'))
        specificity.append(cm_result[0] / (cm_result[0] + cm_result[1]))
        npv.append(cm_result[0] / (cm_result[0] + cm_result[2]))
        fdr.append(cm_result[1] / (cm_result[1] + cm_result[3]))
        accuracy.append(accuracy_score(y_trues[fold], y_preds[fold]))

    tn_mean = np.mean(tn)
    fp_mean = np.mean(fp)
    fn_mean = np.mean(fn)
    tp_mean = np.mean(tp)
    pred_pos_mean = np.mean(pred_pos)
    pred_neg_mean = np.mean(pred_neg)
    known_pos_mean = np.mean(known_pos)
    known_neg_mean = np.mean(known_neg)
    f1_mean = np.mean(f1)
    precision_mean = np.mean(precision)
    recall_mean = np.mean(recall)
    specificity_mean = np.mean(specificity)
    npv_mean = np.mean(npv)
    fdr_mean = np.mean(fdr)
    accuracy_mean = np.mean(accuracy)

    tn_std = np.std(tn)
    fp_std = np.std(fp)
    fn_std = np.std(fn)
    tp_std = np.std(tp)
    pred_pos_std = np.std(pred_pos)
    pred_neg_std = np.std(pred_neg)
    known_pos_std = np.std(known_pos)
    known_neg_std = np.std(known_neg)
    f1_std = np.std(f1)
    precision_std = np.std(precision)
    recall_std = np.std(recall)
    specificity_std = np.std(specificity)
    npv_std = np.std(npv)
    fdr_std = np.std(fdr)
    accuracy_std = np.std(accuracy)

    log.info('Confusion matrix (tp, fp, fn, tn): (%.2f±%.2f, %.2f±%.2f, %.2f±%.2f, %.2f±%.2f)',
             tp_mean, tp_std, fp_mean, fp_std, fn_mean, fn_std, tn_mean, tn_std)
    log.info('pred pos: %.2f±%.2f', pred_pos_mean, pred_pos_std)
    log.info('pred neg: %.2f±%.2f', pred_neg_mean, pred_neg_std)
    log.info('known pos: %.2f±%.2f', known_pos_mean, known_pos_std)
    log.info('known neg: %.2f±%.2f', known_neg_mean, known_neg_std)
    log.info('F1: %.2f±%.2f', f1_mean, f1_std)
    log.info('Precision: %.2f±%.2f', precision_mean, precision_std)
    log.info('Recall: %.2f±%.2f', recall_mean, recall_std)
    log.info('Specificity: %.2f±%.2f', specificity_mean, specificity_std)
    log.info('Npv: %.2f±%.2f', npv_mean, npv_std)
    log.info('Fdr: %.2f±%.2f', fdr_mean, fdr_std)
    log.info('Accuracy: %.2f±%.2f', accuracy_mean, accuracy_std)

    # plot PR curve
    fig = plt.figure()

    lines = []
    labels = []
    for classifier, ys in classifiers_ys.items():
        y_trues, y_preds, y_probs = ys

        if classifier == best_classifier:
            num_folds = len(y_trues)
            precision = 0
            recall = 0

            for fold in range(num_folds):
                precision += precision_score(y_trues[fold], y_preds[fold], average='binary')
                recall += recall_score(y_trues[fold], y_preds[fold], average='binary')

            precision /= num_folds
            recall /= num_folds

            arrowprops = {'arrowstyle': '->'}
            plt.scatter(recall, precision, s=30, marker='x', c='k', zorder=3)
            plt.annotate(
                'Operational point',
                (recall, precision),
                (recall-0.05, precision+0.05),
                arrowprops=arrowprops)

        y_probs_1 = tuple(y_prob[1].to_numpy() for y_prob in y_probs)
        line, label = plot_pr(y_trues, y_probs_1, classifier)

        lines.append(line)
        labels.append(label)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Best model ({}) PR curve'.format(best_classifier))
    plt.legend(lines, labels, loc='upper right', prop={'size': 10})

    save_figure(fig, main_config.get_str('pr_curve'))


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

    # baseline / best classifiers
    baseline_classifier = main_config.get_str('baseline')
    best_classifier = main_config.get_str('classifier')

    # plot PR curve and print confusion matrix
    plot_pr_print_cm(baseline_classifier, best_classifier, main_config, model_manager)


if __name__ == '__main__':
    main()

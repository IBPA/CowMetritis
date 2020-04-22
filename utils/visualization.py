# standard imports
import logging as log
import os

# third party imports
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
import seaborn as sns


def save_figure(fig, save_to):
    fig.savefig(save_to, bbox_inches='tight')


def plot_bic_fs(feature_rank, score_avgs, bic_avgs, save_to):
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(feature_rank, score_avgs, 'g-')
    ax2.plot(feature_rank, bic_avgs, 'b-')

    ax1.set_xlabel('Features')
    ax1.set_ylabel('F1-score', color='g')
    ax2.set_ylabel('BIC', color='b')

    ax1.set_xticklabels(feature_rank, rotation=90)

    save_figure(fig, save_to)


def plot_sfs(metric_dict,
             xticks,
             title,
             save_to,
             filename='sfs.png',
             kind='std_dev',
             color='blue',
             bcolor='steelblue',
             marker='o',
             alpha=0.2,
             confidence_interval=0.95):
    allowed = {'std_dev', 'std_err', 'ci', None}
    if kind not in allowed:
        raise AttributeError('kind not in %s' % allowed)

    plt.figure()

    k_feat = sorted(metric_dict.keys())
    avg = [metric_dict[k]['avg_score'] for k in k_feat]

    if kind:
        upper, lower = [], []
        if kind == 'ci':
            kind = 'ci_bound'

        for k in k_feat:
            upper.append(metric_dict[k]['avg_score'] +
                         metric_dict[k][kind])
            lower.append(metric_dict[k]['avg_score'] -
                         metric_dict[k][kind])

        plt.fill_between(k_feat,
                         upper,
                         lower,
                         alpha=alpha,
                         color=bcolor,
                         lw=1)

        if kind == 'ci_bound':
            kind = 'Confidence Interval (%d%%)' % (confidence_interval * 100)

    plt.plot(k_feat, avg, color=color, marker=marker)
    plt.xlabel('Features')
    plt.ylabel('F1-score')
    plt.title(title)
    plt.grid()
    plt.xticks(range(1, len(xticks)+1), xticks, rotation=90)

    save_figure(plt.gcf(), os.path.join(save_to, filename))


def plot_feature_elimination(x, y, err, save_to, filename='rfe.png'):
    plt.figure()
    plt.errorbar(x, y, yerr=err, elinewidth=0.3, capsize=3, capthick=0.3)
    plt.xlabel('Eliminated feature at each turn')
    plt.ylabel('F1')
    plt.xticks(range(len(x)), x, rotation=90)

    save_figure(plt.gcf(), os.path.join(save_to, filename))


def plot_pairwise_corr(corr, save_to, filename='pairwise_corr.png'):
    plt.figure()
    sns.heatmap(corr, cmap=plt.cm.bwr)
    plt.title('Pairwise Pearson Correlation')
    save_figure(plt.gcf(), os.path.join(save_to, filename))


def visualize_missing_values(pd_data, save_to):
    if not save_to:
        log.warn('No directory specified. Not saving missing value visualization!')
        return

    save_figure(
        msno.matrix(pd_data).get_figure(),
        os.path.join(save_to, 'mv_matrix.png'))


def plot_projection(X, y, mode, save_to, outlier_index=None):
    if not save_to:
        log.warn('No directory specified. Not saving projection!')
        return

    if X.shape[1] == 2:
        projection = None
    elif X.shape[1] == 3:
        projection = '3d'
    else:
        raise ValueError('Invalid shape!')

    if mode.lower() == 'pca':
        name = 'pc'
    elif mode.lower() == 'sparsepca':
        name = 'pc'
    elif mode.lower() == 'tsne':
        name = 'embedding'
    else:
        raise ValueError('Invalid mode: {}'.format(mode))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=projection)
    ax.set_xlabel('{}1'.format(name))
    ax.set_ylabel('{}2'.format(name))
    if X.shape[1] == 3:
        ax.set_zlabel('{}3'.format(name))

    labels = list(set(y.tolist()))
    colors = ['b', 'r']

    for label, color in zip(labels, colors):
        if X.shape[1] == 2:
            ax.scatter(
                X.loc[y == label, '{}1'.format(name)],
                X.loc[y == label, '{}2'.format(name)],
                c=color,
                s=30)
        else:
            ax.scatter(
                X.loc[y == label, '{}1'.format(name)],
                X.loc[y == label, '{}2'.format(name)],
                X.loc[y == label, '{}3'.format(name)],
                c=color,
                s=30)

    if outlier_index:
        X_index = X.index.tolist()
        outliers = [X_index[i] for i, is_outlier in enumerate(outlier_index) if not is_outlier]

        if len(outliers) > 0:
            labels += ['Outlier']

            if X.shape[1] == 2:
                ax.scatter(
                    X.loc[outliers, '{}1'.format(name)],
                    X.loc[outliers, '{}2'.format(name)],
                    c='k',
                    marker='x',
                    s=60)
            else:
                ax.scatter(
                    X.loc[outliers, '{}1'.format(name)],
                    X.loc[outliers, '{}2'.format(name)],
                    X.loc[outliers, '{}3'.format(name)],
                    c='k',
                    marker='x',
                    s=60)

    labels = ['Cured' if label == 1 else label for label in labels]
    labels = ['Not Cured' if label == 0 else label for label in labels]

    ax.legend(labels)
    ax.grid()

    save_figure(fig, os.path.join(save_to, '{}.png'.format(mode)))


def plot_scatter_matrix(X, y, save_to, filename='scatter_matrix.png'):
    if not save_to:
        log.warn('No directory specified. Not saving scatter matrix!')
        return

    color_labels = y.copy()
    color_labels.replace(0, 'red', inplace=True)
    color_labels.replace(1, 'blue', inplace=True)

    color_labels.replace('low', 'red', inplace=True)
    color_labels.replace('high', 'blue', inplace=True)

    axs = scatter_matrix(X, figsize=(20, 20), marker='o', c=color_labels, diagonal='hist')
    fig = axs[0, 0].get_figure()

    save_figure(fig, os.path.join(save_to, 'scatter_matrix.png'))


def plot_pr(y_trues, y_probs, classifier, plot_only_avg=True):
    assert len(y_trues) == len(y_probs)

    num_folds = len(y_trues)

    precision = {}
    recall = {}
    ap = {}
    for fold in range(num_folds):
        precision[fold], recall[fold], _ = precision_recall_curve(y_trues[fold], y_probs[fold])
        ap[fold] = average_precision_score(y_trues[fold], y_probs[fold])

    combined_trues = np.concatenate(y_trues)
    combined_probs = np.concatenate(y_probs)

    avg_precision, avg_recall, _ = precision_recall_curve(combined_trues, combined_probs)

    # do plot
    # if not plot_only_avg:
    #     for fold in range(num_folds):
    #         plt.step(recall[fold], precision[fold], where='post', color='r')

    if classifier == 'DummyClassifier':
        line, = plt.step(avg_recall, avg_precision, linestyle='--', color='k', where='post')
        classifier = 'Baseline'
    else:
        line, = plt.step(avg_recall, avg_precision, where='post', zorder=1)

    label = '{} (mAP: {:.2f})'.format(classifier, np.mean(list(ap.values())))

    return line, label


def plot_roc(y_trues, y_probs, classifier):
    assert len(y_trues) == len(y_probs)

    combined_trues = np.concatenate(y_trues)
    combined_probs = np.concatenate(y_probs)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, thresholds = roc_curve(combined_trues, combined_probs)
    roc_auc = roc_auc_score(combined_trues, combined_probs)

    if classifier == 'DummyClassifier':
        line, = plt.plot(fpr, tpr, linestyle='--', color='k')
        classifier = 'Baseline'
    else:
        line, = plt.plot(fpr, tpr)

    label = '{} (auROC: {:.2f})'.format(classifier, roc_auc)

    return line, label

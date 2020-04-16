# standard imports
import os

# third party imports
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score


def save_figure(fig, save_to):
    fig.savefig(save_to, bbox_inches='tight')


def plot_feature_elimination(x, y, save_to, filename='rfe.png'):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('Eliminated feature at each turn')
    plt.ylabel('F1')
    plt.xticks(range(len(x)), x, rotation=90)

    save_figure(plt.gcf(), os.path.join(save_to, filename))


def plot_feature_importance(
        X,
        importances_avg,
        importances_std,
        indices,
        ranking,
        save_to,
        filename='feature_importance.png'):
    plt.figure()
    plt.xlabel('Features')
    plt.ylabel('Importances')
    plt.bar(
        range(X.shape[1]),
        importances_avg[indices],
        color='r',
        yerr=importances_std[indices],
        align="center")
    plt.xticks(range(X.shape[1]), ranking, rotation=90)
    plt.xlim([-1, X.shape[1]])

    save_figure(plt.gcf(), os.path.join(save_to, filename))


def visualize_missing_values(pd_data, save_to):
    save_figure(
        msno.matrix(pd_data).get_figure(),
        os.path.join(save_to, 'mv_matrix.png'))

    save_figure(
        msno.bar(pd_data).get_figure(),
        os.path.join(save_to, 'mv_bar.png'))

    save_figure(
        msno.heatmap(pd_data).get_figure(),
        os.path.join(save_to, 'mv_heatmap.png'))

    save_figure(
        msno.dendrogram(pd_data).get_figure(),
        os.path.join(save_to, 'mv_dendrogram.png'))


def plot_projection(X, y, mode, save_to, outlier_index=None):
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

    ax.legend(labels)
    ax.grid()

    save_figure(fig, os.path.join(save_to, '{}.png'.format(mode)))


def plot_scatter_matrix(X, y, save_to, filename='scatter_matrix.png'):
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

    line, = plt.step(avg_recall, avg_precision, where='post')
    label = '{} (mAP: {:.2f})'.format(classifier, np.mean(list(ap.values())))

    return line, label


def plot_roc(y_trues, y_probs, classifier):
    assert len(y_trues) == len(y_probs)

    combined_trues = np.concatenate(y_trues)
    combined_probs = np.concatenate(y_probs)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, thresholds = roc_curve(combined_trues, combined_probs)
    roc_auc = roc_auc_score(combined_trues, combined_probs)

    line, = plt.plot(fpr, tpr)
    label = '{} (auROC: {:.2f})'.format(classifier, roc_auc)

    return line, label

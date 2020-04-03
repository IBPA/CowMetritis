# standard imports
import os

# third party imports
import missingno as msno
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


def save_figure(fig, save_to):
    fig.savefig(save_to, bbox_inches='tight')


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


def plot_scatter_matrix(X, y, save_to):
    color_labels = y.copy()
    color_labels.replace(0, 'red', inplace=True)
    color_labels.replace(1, 'blue', inplace=True)

    color_labels.replace('low', 'red', inplace=True)
    color_labels.replace('high', 'blue', inplace=True)

    axs = scatter_matrix(X, figsize=(20, 20), marker='o', c=color_labels, diagonal='hist')
    fig = axs[0, 0].get_figure()

    save_figure(fig, os.path.join(save_to, 'scatter_matrix.png'))

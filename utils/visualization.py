# standard imports
import os

# third party imports
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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


def plot_projection(X, y, save_to, mode):
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

    # labels = ['treated_cured', 'treated_uncured', 'untreated_cured', 'untreated_uncured']
    # colors = ['r', 'g', 'b', 'y']

    labels = [1, 0]
    colors = ['b', 'r']

    for label, color in zip(labels, colors):
        if X.shape[1] == 2:
            ax.scatter(
                X.loc[y == label, '{}1'.format(name)],
                X.loc[y == label, '{}2'.format(name)],
                c=color,
                s=50)
        else:
            ax.scatter(
                X.loc[y == label, '{}1'.format(name)],
                X.loc[y == label, '{}2'.format(name)],
                X.loc[y == label, '{}3'.format(name)],
                c=color,
                s=50)

    ax.legend(labels)
    ax.grid()

    save_figure(fig, os.path.join(save_to, '{}.png'.format(mode)))

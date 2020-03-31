# standard imports
import os

# third party imports
import missingno as msno
import matplotlib.pyplot as plt

def save_figure(fig, save_to):
	fig.savefig(save_to, bbox_inches='tight')

def visualize_missing_values(pd_data, save_to):
	matrix = msno.matrix(pd_data)
	save_figure(matrix.get_figure(), os.path.join(save_to, 'mv_matrix.pdf'))

	bar = msno.bar(pd_data)
	save_figure(bar.get_figure(), os.path.join(save_to, 'mv_bar.pdf'))

	heatmap = msno.heatmap(pd_data)
	save_figure(heatmap.get_figure(), os.path.join(save_to, 'mv_heatmap.pdf'))

	dendrogram = msno.dendrogram(pd_data)
	save_figure(dendrogram.get_figure(), os.path.join(save_to, 'mv_dendrogram.pdf'))

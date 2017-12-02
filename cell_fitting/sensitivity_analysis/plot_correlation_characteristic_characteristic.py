import os
import numpy as np
from cell_fitting.sensitivity_analysis.plot_correlation_param_characteristic import compute_and_plot_correlations
import types
import matplotlib
import matplotlib.pyplot as pl
pl.style.use('paper')


def plot_corr(corr, sig_level, y_tick_labels, x_tick_labels, correlation_type, save_dir_plots):
    corr = np.flip(corr, 0)  # to have diagonal from upper left to lower right
    pl.figure()
    pl.pcolor(corr, vmin=-1, vmax=1, cmap=pl.cm.get_cmap('gray'))
    # mark nans
    corr_nan = np.ma.masked_where(~np.isnan(corr), corr)
    corr_nan[np.isnan(corr_nan)] = 0.5
    h = pl.pcolor(corr_nan, vmin=-1, vmax=1, cmap=pl.cm.get_cmap('gray'))
    h.set_hatch('\\\\\\\\')
    pl.xlabel('Characteristic')
    pl.ylabel('Characteristic')
    pl.xticks(np.arange(len(x_tick_labels)) + 0.5, x_tick_labels, rotation='40', ha='right')
    pl.yticks(np.arange(len(y_tick_labels)) + 0.5, y_tick_labels[::-1])
    ax = pl.gca()
    for label in ax.xaxis.get_majorticklabels():
        label.customShiftValue = -0.2
        label.set_x = types.MethodType(lambda self, x: matplotlib.text.Text.set_x(self, x - self.customShiftValue),
                                       label, matplotlib.text.Text)
    ax.tick_params(axis='x', which='major', pad=0)
    pl.axis('scaled')
    cb = pl.colorbar(fraction=0.1)
    cb.set_label(correlation_type + ' correlation', rotation=-90, labelpad=30)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_plots, correlation_type + '_p'+ str(sig_level) + '.png'))
    #pl.show()


if __name__ == '__main__':
    # save dir
    save_dir_analysis = os.path.join('../results/sensitivity_analysis', 'mean_2std_6models', 'analysis')
    save_dir_plots = os.path.join(save_dir_analysis, 'plots', 'correlation', 'characteristics_characteristic', 'all')

    correlation_types = ['kendalltau', 'spearman', 'pearson']  # 'spearman  # 'kendalltau  # 'pearson'
    sig1 = 0.01
    sig2 = 0.001

    # load
    return_characteristics = np.load(os.path.join(save_dir_analysis, 'return_characteristics.npy'))
    characteristics_mat = np.load(os.path.join(save_dir_analysis, 'characteristics_mat.npy'))

    if not os.path.exists(save_dir_plots):
        os.makedirs(save_dir_plots)

    compute_and_plot_correlations(characteristics_mat, characteristics_mat, correlation_types,
                                  sig1, sig2, return_characteristics, return_characteristics, plot_corr,
                                  save_dir_plots)
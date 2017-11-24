import os
import json
import numpy as np
from cell_fitting.sensitivity_analysis import rename_nat_and_nap
from cell_fitting.sensitivity_analysis.plot_correlation_param_characteristic import compute_and_plot_correlations
from cell_fitting.optimization.helpers import get_lowerbound_upperbound_keys
import types
import matplotlib
import matplotlib.pyplot as pl
pl.style.use('paper')


def plot_corr(corr, sig_level, y_tick_labels, x_tick_labels, correlation_type, save_dir_plots):
    corr = np.flip(corr, 0)
    pl.figure(figsize=(10, 7.8))
    pl.pcolor(corr, vmin=-1, vmax=1, cmap=pl.cm.get_cmap('gray'))
    # mark nans
    corr_nan = np.ma.masked_where(~np.isnan(corr), corr)
    corr_nan[np.isnan(corr_nan)] = 0.5
    h = pl.pcolor(corr_nan, vmin=-1, vmax=1, cmap=pl.cm.get_cmap('gray'))
    h.set_hatch('\\\\\\\\')
    pl.xlabel('Parameter')
    pl.ylabel('Parameter')
    pl.xticks(np.arange(len(x_tick_labels)) + 0.5, x_tick_labels, rotation='40', ha='right', fontsize=9)
    pl.yticks(np.arange(len(y_tick_labels)) + 0.5, y_tick_labels[::-1], fontsize=9)
    ax = pl.gca()
    for label in ax.xaxis.get_majorticklabels():
        label.customShiftValue = -0.3
        label.set_x = types.MethodType(lambda self, x: matplotlib.text.Text.set_x(self, x - self.customShiftValue),
                                       label, matplotlib.text.Text)
    ax.tick_params(axis='x', which='major', pad=0)
    pl.axis('scaled')
    cb = pl.colorbar(fraction=0.045)
    cb.set_label(correlation_type + ' correlation', rotation=-90, labelpad=30)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_plots, correlation_type + '_p' + str(sig_level) + '.png'))
    #pl.show()


def get_candidates_in_range(range_candidates, candidate_idxs, candidate_mat, characteristics_mat):
    lower_bounds, upper_bounds, characteristics = get_lowerbound_upperbound_keys(range_candidates)

    candidates_in_range = []
    for candidate_idx, candidate, candidate_characteristics in zip(candidate_idxs, candidate_mat, characteristics_mat):
        if np.all(lower_bounds < candidate_characteristics) and np.all(candidate_characteristics < upper_bounds):
            candidates_in_range.append((candidate_idx, candidate))
    return candidates_in_range

if __name__ == '__main__':
    # save dir
    range_name = 'all'
    save_dir_analysis = os.path.join('../results/sensitivity_analysis/', 'mean_std_6models', 'analysis')
    save_dir_plots = os.path.join(save_dir_analysis, 'plots', 'correlation', 'parameter_parameter', range_name)

    correlation_types = ['kendalltau', 'spearman', 'pearson']
    sig1 = 0.01
    sig2 = 0.001

    # load
    with open(os.path.join(save_dir_analysis, 'params.json'), 'r') as f:
        params = json.load(f)
    variable_names = [p[2][0][-2] + ' ' + p[2][0][-1] for p in params['variables']]
    variable_names = rename_nat_and_nap(variable_names)
    candidate_mat = np.load(os.path.join(save_dir_analysis, 'candidate_mat.npy'))
    n_variables = np.shape(candidate_mat)[1]
    if not os.path.exists(save_dir_plots):
        os.makedirs(save_dir_plots)

    # select candidates with certain DAPs
    range_candidates = [
        [50, 150, 'AP_amp'],
        [0.1, 2.0, 'AP_width'],
        [0, 40, 'fAHP_amp'],
        [0, 40, 'DAP_amp'],
        [0, 20, 'DAP_deflection'],
        [0, 70, 'DAP_width'],
        [0, 20, 'DAP_time'],
        #[-np.inf, np.inf, 'DAP_lin_slope'],
        #[-np.inf, np.inf, 'DAP_exp_slope']
    ]
    # [50, 150, 'AP_amp'],
    # [0.1, 2.0, 'AP_width'],
    # [0, 40, 'fAHP_amp'],
    # [0, 40, 'DAP_amp'],
    # [0, 20, 'DAP_deflection'],
    # [0, 70, 'DAP_width'],
    # [0, 20, 'DAP_time'],
    # [-np.inf, np.inf, 'DAP_lin_slope'],
    # [-np.inf, np.inf, 'DAP_exp_slope']
    characteristics_mat = np.load(os.path.join(save_dir_analysis, 'characteristics_mat.npy'))
    candidate_idxs = np.load(os.path.join(save_dir_analysis, 'candidate_idxs.npy'))
    candidates_in_range = get_candidates_in_range(range_candidates, candidate_idxs, candidate_mat, characteristics_mat)
    candidate_mat_range = np.zeros((len(candidates_in_range), n_variables))
    for i, (candidate_idx, candidate) in enumerate(candidates_in_range):
        candidate_mat_range[i] = candidate

    compute_and_plot_correlations(candidate_mat_range, candidate_mat_range, correlation_types, sig1, sig2,
                                  variable_names, variable_names, plot_corr, save_dir_plots)
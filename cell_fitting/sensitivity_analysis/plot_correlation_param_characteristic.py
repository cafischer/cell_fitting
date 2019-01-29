import os
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from cell_fitting.util import init_nan
from cell_fitting.sensitivity_analysis import rename_nat_and_nap
import types
import matplotlib
import matplotlib.pyplot as pl
from cell_fitting.util import characteristics_dict_for_plotting, get_variable_names_for_plotting
from statsmodels.sandbox.stats.multicomp import multipletests
pl.style.use('paper')


def get_divisors(x):
    divisors = []
    for i in range(1, int(x**0.5)+1):
        if x % i == 0:
            divisors.append((i, x / i))
    return divisors


def compute_correlation_matrix(candidate_mat, characteristics_mat, correlation_type):
    n_variables = np.shape(candidate_mat)[1]
    n_characteristics = np.shape(characteristics_mat)[1]
    corr_mat = init_nan((n_characteristics, n_variables))
    p_val = init_nan((n_characteristics, n_variables))
    for i_characteristic in range(n_characteristics):
        not_nan = np.logical_not(np.isnan(characteristics_mat[:, i_characteristic]))
        for i_var in range(n_variables):
            if not np.sum(not_nan) == 0:
                if correlation_type == 'pearson':
                    corr_mat[i_characteristic, i_var], p_val[i_characteristic, i_var] = pearsonr(
                        candidate_mat[not_nan, i_var],
                        characteristics_mat[not_nan, i_characteristic])
                elif correlation_type == 'spearman':
                    corr_mat[i_characteristic, i_var], p_val[i_characteristic, i_var] = spearmanr(
                        candidate_mat[not_nan, i_var],
                        characteristics_mat[not_nan, i_characteristic])
                elif correlation_type == 'kendalltau':
                    corr_mat[i_characteristic, i_var], p_val[i_characteristic, i_var] = kendalltau(
                        candidate_mat[not_nan, i_var],
                        characteristics_mat[not_nan, i_characteristic])
    return corr_mat, p_val


def plot_corr(corr, sig_level, return_characteristics, variable_names, correlation_type, save_dir_plots):
    pl.figure(figsize=(12, 3.7))
    pl.pcolor(corr, vmin=-1, vmax=1, cmap=pl.cm.get_cmap('gray'))
    # mark nans
    corr_nan = np.ma.masked_where(~np.isnan(corr), corr)
    corr_nan[np.isnan(corr_nan)] = 0.5
    h = pl.pcolor(corr_nan, vmin=-1, vmax=1, cmap=pl.cm.get_cmap('gray'))
    h.set_hatch('\\\\\\\\')
    pl.xlabel('Parameter')
    pl.ylabel('Characteristic')
    pl.xticks(np.arange(len(variable_names)) + 0.5, variable_names, rotation='40', ha='right', fontsize=9)
    pl.yticks(np.arange(len(return_characteristics)) + 0.5, return_characteristics, fontsize=9)
    ax = pl.gca()
    for label in ax.xaxis.get_majorticklabels():
        label.customShiftValue = -0.3
        label.set_x = types.MethodType(lambda self, x: matplotlib.text.Text.set_x(self, x - self.customShiftValue),
                                       label, matplotlib.text.Text)
    ax.tick_params(axis='x', which='major', pad=0)
    pl.axis('scaled')
    cb = pl.colorbar(fraction=0.009)
    cb.set_label(correlation_type + ' correlation', rotation=-90, labelpad=30)
    pl.tight_layout()
    pl.subplots_adjust(top=0.94, bottom=0.25)
    pl.savefig(os.path.join(save_dir_plots, correlation_type + '_p' + str(sig_level) + '.png'))
    #pl.show()


def plot_corr_on_ax(ax, corr_mat, p_val_mat, return_characteristics, variable_names, correlation_measure, units,
                    cmap='gray'):
    X, Y = np.meshgrid(np.arange(np.size(corr_mat, 1)+1), np.arange(np.size(corr_mat, 0)+1))  # +1 because otherwise pcolor misses the last row
    pl.pcolor(X, Y, np.flipud(corr_mat), vmin=-1, vmax=1, cmap=pl.cm.get_cmap(cmap))

    # p-values
    sig_levels = [0.1, 0.01, 0.001, 0]
    markers = ['*', '**', '***']
    for i in range(len(sig_levels)-1):
        _, p_val_corrected, _, _ = multipletests(p_val_mat.flatten(), sig_levels[i], method='bonferroni')
        p_val_corrected = np.reshape(p_val_corrected, np.shape(p_val_mat))

        for i, sig in enumerate(sig_levels[:-1]):
            reject = np.logical_and(p_val_corrected < sig_levels[i], ~(p_val_corrected < sig_levels[i+1]))

            ys, xs = np.where(reject)
            for x, y in zip(xs, ys):
                ax.annotate(markers[i], xy=(x + 0.5, y + 0.5), ha='center', va='center', fontsize=10)



    ax.set_xlabel('Parameter')
    ax.set_ylabel('Characteristic')
    ax.set_xticks(np.arange(len(variable_names)) + 0.5)
    new_variable_names = get_variable_names_for_plotting(variable_names)
    ax.set_xticklabels(new_variable_names, rotation='40', ha='right')
    ax.set_yticks(np.arange(len(return_characteristics)) + 0.5)
    characteristics_dict = characteristics_dict_for_plotting()
    ax.set_yticklabels([characteristics_dict[c] + ' ('+units[len(units)-i-1]+')'
                        for i, c in enumerate(return_characteristics[::-1])])  # flipped with pcolor
    ax = pl.gca()
    for label in ax.xaxis.get_majorticklabels():
        label.customShiftValue = -0.3
        label.set_x = types.MethodType(lambda self, x: matplotlib.text.Text.set_x(self, x - self.customShiftValue),
                                       label, matplotlib.text.Text)
    ax.tick_params(axis='x', which='major', pad=0)
    ax.axis('scaled')
    cb = pl.colorbar(fraction=0.0055, ticks=[-1, -0.5, 0, 0.5, 1])
    if correlation_measure == 'kendalltau':
        correlation_measure = 'Kendalls tau'
    elif correlation_measure == 'spearman':
        correlation_measure = 'Spearman'
    elif correlation_measure == 'pearson':
        correlation_measure = 'Pearson'
    cb.set_label(correlation_measure, rotation=-90, labelpad=13)
    cb.ax.set_yticklabels([-1.0, -0.5, 0.0, 0.5, 1.0])


def compute_and_plot_correlations(candidate_mat, characteristics_mat, correlation_types, sig1, sig2,
                                  variable_names, return_characteristics, plot_corr, save_dir_plots=None):
    for correlation_type in correlation_types:
        # compute correlation matrices
        corr_mat, p_val = compute_correlation_matrix(candidate_mat, characteristics_mat, correlation_type)
        if save_dir_plots is not None:
            np.save(os.path.join(save_dir_plots, 'correlation_' + correlation_type + '.npy'), corr_mat)
            np.save(os.path.join(save_dir_plots, 'p_val_' + correlation_type + '.npy'), p_val)
            np.save(os.path.join(save_dir_plots, 'return_characteristics_' + correlation_type + '.npy'),
                    return_characteristics)
            np.save(os.path.join(save_dir_plots, 'variable_names_' + correlation_type + '.npy'), variable_names)

        # plot feature vs parameter + color = correlation
        p_sig1 = p_val < sig1
        p_sig2 = p_val < sig2
        corr_p_sig1 = init_nan(np.shape(corr_mat))
        corr_p_sig1[p_sig1] = corr_mat[p_sig1]
        corr_p_sig2 = init_nan(np.shape(corr_mat))
        corr_p_sig2[p_sig2] = corr_mat[p_sig2]

        if save_dir_plots is not None:
            plot_corr(corr_mat, 1, return_characteristics, variable_names, correlation_type, save_dir_plots)
            plot_corr(corr_p_sig1, sig1, return_characteristics, variable_names, correlation_type, save_dir_plots)
            plot_corr(corr_p_sig2, sig2, return_characteristics, variable_names, correlation_type, save_dir_plots)

            # scatter plots
            if not os.path.exists(os.path.join(save_dir_plots, 'scatter')):
                os.makedirs(os.path.join(save_dir_plots, 'scatter'))
            for i_c, c in enumerate(return_characteristics):
                for i_p, p in enumerate(variable_names):
                    pl.figure()
                    pl.plot(candidate_mat[:, i_p], characteristics_mat[:, i_c], 'ok')
                    pl.xlabel(p)
                    pl.ylabel(c)
                    pl.tight_layout()
                    pl.savefig(os.path.join(save_dir_plots, 'scatter', c+'_'+p+'.png'))
                    #pl.show()
                    pl.close()

    return corr_mat, corr_p_sig1, corr_p_sig2

if __name__ == '__main__':
    # save dir
    save_dir = os.path.join('../results/sensitivity_analysis/', 'mean_std_1order_of_mag_model2')
    save_dir_analysis = os.path.join(save_dir, 'analysis')
    save_dir_plots = os.path.join(save_dir_analysis, 'plots', 'correlation', 'parameter_characteristic', 'all')
    characteristics = ['DAP_deflection', 'DAP_amp', 'DAP_width', 'DAP_time']
    correlation_types = ['kendalltau', 'spearman', 'pearson']  # 'spearman  # 'kendalltau  # 'pearson'
    sig1 = 0.01
    sig2 = 0.001

    # load
    with open(os.path.join(save_dir, 'params.json'), 'r') as f:
        params = json.load(f)
    variable_names = [p[2][0][-2] + ' ' + p[2][0][-1] for p in params['variables']]
    variable_names = rename_nat_and_nap(variable_names)
    return_characteristics = np.load(os.path.join(save_dir_analysis, 'return_characteristics.npy'))
    characteristics_mat = np.load(os.path.join(save_dir_analysis, 'characteristics_mat.npy'))
    candidate_mat = np.load(os.path.join(save_dir_analysis, 'candidate_mat.npy'))
    n_variables = np.shape(candidate_mat)[1]
    if not os.path.exists(save_dir_plots):
        os.makedirs(save_dir_plots)
    characteristic_idxs = np.array([np.where(characteristic == return_characteristics)[0][0]
                                    for characteristic in characteristics], dtype=int)

    compute_and_plot_correlations(candidate_mat, characteristics_mat[:, characteristic_idxs], correlation_types, sig1, sig2,
                                  variable_names, return_characteristics[characteristic_idxs], plot_corr, save_dir_plots)
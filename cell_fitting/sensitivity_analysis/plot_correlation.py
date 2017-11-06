import os
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from cell_fitting.util import init_nan
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as pl
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
    pl.figure(figsize=(12, 4.0))
    pl.pcolor(corr, vmin=-1, vmax=1)
    pl.xlabel('Parameter')
    pl.ylabel('Characteristic')
    pl.xticks(np.arange(len(variable_names)) + 0.5, variable_names, rotation='40', ha='right')
    pl.yticks(np.arange(len(return_characteristics)) + 0.5, return_characteristics)
    pl.axis('scaled')
    cb = pl.colorbar(fraction=0.01)
    cb.set_label(correlation_type + ' correlation', rotation=-90, labelpad=30)
    pl.tight_layout()
    pl.subplots_adjust(top=1.0, bottom=0.36)
    pl.savefig(os.path.join(save_dir_plots, 'feature_parameter_correlation_' + correlation_type + '_p'
                            + str(sig_level) + '.png'))
    #pl.show()


def compute_and_plot_correlations(candidate_mat, characteristics_mat, correlation_types, sig1, sig2,
                                  variable_names, return_characteristics, save_dir_plots=None):
    for correlation_type in correlation_types:
        # compute correlation matrices
        corr_mat, p_val = compute_correlation_matrix(candidate_mat, characteristics_mat, correlation_type)
        if save_dir_plots is not None:
            np.save(os.path.join(save_dir_plots, 'correlation_' + correlation_type + '.npy'), corr_mat)
            np.save(os.path.join(save_dir_plots, 'p_val_' + correlation_type + '.npy'), p_val)

        # plot feature vs parameter + color = correlation
        p_sig1 = p_val < sig1
        p_sig2 = p_val < sig2
        corr_masked = np.ma.masked_where(np.isnan(corr_mat), corr_mat)
        corr_p_sig1_masked = np.ma.masked_where(np.logical_not(p_sig1), corr_mat)
        corr_p_sig2_masked = np.ma.masked_where(np.logical_not(p_sig2), corr_mat)

        if save_dir_plots is not None:
            plot_corr(corr_masked, 1, return_characteristics, variable_names, correlation_type, save_dir_plots)
            plot_corr(corr_p_sig1_masked, sig1, return_characteristics, variable_names, correlation_type, save_dir_plots)
            plot_corr(corr_p_sig2_masked, sig2, return_characteristics, variable_names, correlation_type, save_dir_plots)

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

    return corr_masked, corr_p_sig1_masked, corr_p_sig2_masked


if __name__ == '__main__':
    # save dir
    save_dir_analysis = os.path.join('../results/sensitivity_analysis/', 'analysis_2017-10-10')
    save_dir_plots = os.path.join(save_dir_analysis, 'plots', 'correlation', 'all')

    correlation_types = ['kendalltau', 'spearman', 'pearson']  # 'spearman  # 'kendalltau  # 'pearson'
    sig1 = 0.01
    sig2 = 0.001

    # load
    with open(os.path.join(save_dir_analysis, 'params.json'), 'r') as f:
        params = json.load(f)
    variable_names = [p[2][0][-2] + ' ' + p[2][0][-1] for p in params['variables']]
    return_characteristics = np.load(os.path.join(save_dir_analysis, 'return_characteristics.npy'))
    characteristics_mat = np.load(os.path.join(save_dir_analysis, 'characteristics_mat.npy'))
    candidate_mat = np.load(os.path.join(save_dir_analysis, 'candidate_mat.npy'))
    n_variables = np.shape(candidate_mat)[1]
    if not os.path.exists(save_dir_plots):
        os.makedirs(save_dir_plots)

    compute_and_plot_correlations(candidate_mat, characteristics_mat[:, :-2], correlation_types, sig1, sig2,
                                  variable_names, return_characteristics[:-2], save_dir_plots)
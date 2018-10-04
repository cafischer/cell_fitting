from __future__ import division
import os
import json
import numpy as np
from cell_fitting.sensitivity_analysis.plot_correlation_param_characteristic import compute_and_plot_correlations, \
    plot_corr
from itertools import product
from cell_fitting.sensitivity_analysis import rename_nat_and_nap
import matplotlib.pyplot as pl
pl.style.use('paper')

# save dir
save_dir_analysis = os.path.join('../results/sensitivity_analysis/', 'mean_std_1order_of_mag_model2', 'analysis')
save_dir_plots = os.path.join(save_dir_analysis, 'plots', 'correlation', 'parameter_characteristic', 'sampled')

n_chunks = 100
correlation_type = 'kendalltau' # 'spearman', 'pearson'
sig1 = 0.01
sig2 = 0.001

# load
with open(os.path.join(save_dir_analysis, 'params.json'), 'r') as f:
    params = json.load(f)
variable_names = [p[2][0][-2] + ' ' + p[2][0][-1] for p in params['variables']]
variable_names = rename_nat_and_nap(variable_names)
return_characteristics = np.load(os.path.join(save_dir_analysis, 'return_characteristics.npy'))
characteristics_mat = np.load(os.path.join(save_dir_analysis, 'characteristics_mat.npy'))
candidate_mat = np.load(os.path.join(save_dir_analysis, 'candidate_mat.npy'))
n_variables = np.shape(candidate_mat)[1]

# chunk up matrices
size_chunk = int(np.floor(np.shape(candidate_mat)[0] / n_chunks))
correlation_mats = np.zeros((len(return_characteristics), n_variables, n_chunks))
for chunk_idx in range(n_chunks):
    #save_dir_chunk = os.path.join(save_dir_plots, 'true', str(chunk_idx))
    #if not os.path.exists(save_dir_chunk):
    #    os.makedirs(save_dir_chunk)
    characteristics_mat_chunk = characteristics_mat[chunk_idx*size_chunk:(chunk_idx+1)*size_chunk, :]
    candidate_mat_chunk = candidate_mat[chunk_idx*size_chunk:(chunk_idx+1)*size_chunk, :]

    corr_masked, _, _ = compute_and_plot_correlations(candidate_mat_chunk, characteristics_mat_chunk, [correlation_type],
                                                      sig1, sig2, variable_names, return_characteristics, plot_corr)
    correlation_mats[:, :, chunk_idx] = corr_masked

# mean and std of correlation mats
mean = np.mean(correlation_mats, 2)
std = np.std(correlation_mats, 2)
variable_characteristics_pairs = np.reshape([c + '-' + v for (c, v) in list(product(return_characteristics, variable_names))],
                                            np.shape(mean))

# save
save_dir_plots = os.path.join(save_dir_plots, correlation_type)
if not os.path.exists(save_dir_plots):
    os.makedirs(save_dir_plots)

np.save(os.path.join(save_dir_plots, 'mean_mat.npy'), mean)
np.save(os.path.join(save_dir_plots, 'std_mat.npy'), std)

# plot
pl.figure(figsize=(19, 7))
pl.errorbar(range(len(variable_characteristics_pairs.flatten())), mean.flatten(), yerr=std.flatten(), fmt='o', color='k',
            capsize=3)
pl.xticks(range(len(variable_characteristics_pairs.flatten())), variable_characteristics_pairs.flatten(), rotation='40',
          ha='right', fontsize=12)
pl.ylim(-1, 1)
pl.tight_layout()
pl.savefig(os.path.join(save_dir_plots, 'mean_std.png'))

for i, c in enumerate(return_characteristics):
    pl.figure(figsize=(14, 5))
    ax = pl.gca()
    ax.errorbar(range(n_variables), mean[i, :], yerr=std[i, :], fmt='o', color='k',
                capsize=3)
    xlim = ax.get_xlim()
    pl.hlines(0.5, *xlim, colors='0.5', linestyles='--')
    pl.hlines(-0.5, *xlim, colors='0.5', linestyles='--')
    pl.xticks(range(n_variables), variable_names, rotation='40', ha='right')
    pl.ylabel(c)
    pl.ylim(-1, 1)
    pl.xlabel('Parameter')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_plots, c+'_mean_std.png'))
    pl.show()
from __future__ import division
import os
import json
import numpy as np
from cell_fitting.sensitivity_analysis.plot_correlation import compute_and_plot_correlations

# save dir
date = '2017-10-26_14:13:11'
save_dir_params = os.path.join('../results/sensitivity_analysis/', date)
save_dir_analysis = os.path.join('../results/sensitivity_analysis/', 'analysis_test')
save_dir_plots = os.path.join(save_dir_analysis, 'plots', 'correlation', 'cross_validation')

n_chunks = 2
correlation_types = ['kendalltau', 'spearman', 'pearson']
sig1 = 0.01
sig2 = 0.001

# load
with open(os.path.join(save_dir_params, 'params.json'), 'r') as f:
    params = json.load(f)
variable_names = [p[2][0][-2] + ' ' + p[2][0][-1] for p in params['variables']]
return_characteristics = np.load(os.path.join(save_dir_analysis, 'return_characteristics.npy'))
characteristics_mat = np.load(os.path.join(save_dir_analysis, 'characteristics_mat.npy'))
candidate_mat = np.load(os.path.join(save_dir_analysis, 'candidate_mat.npy'))
n_variables = np.shape(candidate_mat)[1]

# chunk up matrices
size_chunk = int(np.floor(np.shape(candidate_mat)[0] / n_chunks))
for chunk_idx in range(n_chunks):
    save_dir_chunk = os.path.join(save_dir_plots, str(chunk_idx))
    if not os.path.exists(save_dir_chunk):
        os.makedirs(save_dir_chunk)
    characteristics_mat_chunk = characteristics_mat[chunk_idx*size_chunk:(chunk_idx+1)*size_chunk, :]
    candidate_mat_chunk = candidate_mat[chunk_idx*size_chunk:(chunk_idx+1)*size_chunk, :]

    compute_and_plot_correlations(candidate_mat_chunk, characteristics_mat_chunk, correlation_types, sig1, sig2,
                                  variable_names, return_characteristics, save_dir_chunk)


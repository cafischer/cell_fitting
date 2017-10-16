import os
import json
import numpy as np
from scipy.stats import pearsonr
from itertools import product
from cell_fitting.util import init_nan
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pl
pl.style.use('paper')

def get_divisors(x):
    divisors = []
    for i in range(1, int(x**0.5)+1):
        if x % i == 0:
            divisors.append((i, x / i))
    return divisors


# save dir
dates = ['2017-10-13_08:56:01']
save_dirs = [os.path.join('../results/sensitivity_analysis/', date) for date in dates]
save_dir_analysis = os.path.join('../results/sensitivity_analysis/', 'analysis_1')
save_dir_plots = os.path.join(save_dir_analysis, 'plots')

# load
with open(os.path.join(save_dirs[0], 'params.json'), 'r') as f:
    params = json.load(f)
with open(os.path.join(save_dirs[0], 't.npy'), 'r') as f:
    t = np.load(f)
with open(os.path.join(save_dirs[0], 'i_inj.npy'), 'r') as f:
    i_inj = np.load(f)
with open(os.path.join(save_dir_analysis, 'return_characteristics.npy'), 'r') as f:
    return_characteristics = np.load(f)

# build candidate_mat and characteristics_mat
n_variables = len(params['variables'])
n_total_candidates = params['n_candidates']*len(save_dirs)
candidate_mat = np.zeros((n_total_candidates, n_variables))
characteristics_mat = np.zeros((n_total_candidates, len(return_characteristics)))

for i_dir, save_dir in enumerate(save_dirs):
    for i_candidate in range(params['n_candidates']):
        candidate_dir = os.path.join(save_dir, str(i_candidate))
        with open(os.path.join(candidate_dir, 'candidate.npy'), 'r') as f:
            candidate_mat[i_dir*params['n_candidates']+i_candidate, :] = np.load(f)

        candidate_dir_analysis = os.path.join(save_dir_analysis, dates[i_dir], str(i_candidate))
        with open(os.path.join(candidate_dir_analysis, 'characteristics.npy'), 'r') as f:
            characteristics_mat[i_dir*params['n_candidates']+i_candidate, :] = np.load(f)


# compute correlations
corr_pearson = init_nan((len(return_characteristics), n_variables))
for i_characteristic in range(len(return_characteristics)):
    not_nan = np.logical_not(np.isnan(characteristics_mat[:, i_characteristic]))
    for i_var in range(n_variables):
        if not np.sum(not_nan) == 0:
            corr_pearson[i_characteristic, i_var] = pearsonr(candidate_mat[not_nan, i_var],
                                                         characteristics_mat[not_nan, i_characteristic])[0]

# for i_characteristic in range(len(return_characteristics)):
#
#     # scatter plot
#     # n_plots_hor = get_divisors(n_variables)[-1][0]
#     # n_plots_ver = get_divisors(n_variables)[-1][1]
#     # i = 0
#     # while n_plots_hor < 3:
#     #     i += 1
#     #     n_plots_hor = get_divisors(n_variables+i)[-1][0]
#     #     n_plots_ver = get_divisors(n_variables+i)[-1][1]
#     #
#     # fig, ax = pl.subplots(n_plots_ver, n_plots_hor)
#     # pl.suptitle(return_characteristics[i_characteristic], fontsize=18)
#     # for i, j in product(range(n_plots_ver), range(n_plots_hor)):
#     #     counter = i * n_plots_hor + j
#     #     if counter < n_variables:
#     #         ax[i, j].plot(candidate_mat[:, counter], characteristics_mat[:, i_characteristic], 'ok')
#     #         ax[i, j].plot(candidate_mat[:, counter], corr_pearson[counter] * candidate_mat[:, counter])
#     #         #ax[i, j].set_title('Correlation: '+str(corr_pearson))
#     # #pl.tight_layout()
#     # pl.show()
#
#     # bar plot
#     pl.figure()
#     pl.bar(range(n_variables), corr_pearson[i_characteristic, :], 0.5)
#     pl.xlabel('Parameter')
#     pl.ylabel('Correlation(Parameter, '+return_characteristics[i_characteristic]+')')
#     pl.tight_layout()
#     pl.show()

# plot feature vs parameter + color = correlation
if not os.path.exists(save_dir_plots):
    os.makedirs(save_dir_plots)

variable_names = [p[2][0][-2] + ' ' + p[2][0][-1] for p in params['variables']]
pl.figure(figsize=(11, 4))
corr_masked = np.ma.masked_where(np.isnan(corr_pearson), corr_pearson)
pl.pcolor(corr_masked, vmin=-1, vmax=1)
pl.xlabel('Parameter')
pl.ylabel('Characteristic')
pl.xticks(np.arange(n_variables)+0.5, variable_names, rotation='40', ha='right')
pl.yticks(np.arange(len(return_characteristics))+0.5, return_characteristics)
pl.axis('scaled')
pl.colorbar(fraction=0.01)
pl.tight_layout()
pl.subplots_adjust(top=1.0, bottom=0.45)
pl.savefig(os.path.join(save_dir_plots, 'feature_parameter_correlation.png'))
pl.show()
from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from nrn_wrapper import Cell, load_mechanism_dir
import matplotlib
from cell_fitting.optimization.helpers import get_lowerbound_upperbound_keys
pl.style.use('paper')


model_dirs = [
    '../../results/hand_tuning/cell_2017-07-24_13:59:54_21_0',
    '../../results/hand_tuning/test0/',
    '../../results/server/2017-07-06_13:50:52/434/L-BFGS-B/model',
    '../../results/server/2017-07-17_17:05:19/54/L-BFGS-B/model',
    '../../results/server/2017-07-18_11:14:25/17/L-BFGS-B/model',
    '../../results/server/2017-07-27_09:18:59/22/L-BFGS-B/model'
]
mechanism_dir = '../../model/channels/vavoulis'

variables = [
            [0.3, 1, [['soma', 'cm']]],
            [-90, -80, [['soma', '0.5', 'pas', 'e']]],
            [-30, -10, [['soma', '0.5', 'hcn_slow', 'ehcn']]],

            [0, 0.01, [['soma', '0.5', 'pas', 'g']]],
            [0, 0.3, [['soma', '0.5', 'nat', 'gbar']]],
            [0, 0.3, [['soma', '0.5', 'nap', 'gbar']]],
            [0, 0.3, [['soma', '0.5', 'kdr', 'gbar']]],
            [0, 0.001, [['soma', '0.5', 'hcn_slow', 'gbar']]],

            # [0, 6, [['soma', '0.5', 'nat', 'm_pow']]],
            # [0, 6, [['soma', '0.5', 'nat', 'h_pow']]],
            # [0, 6, [['soma', '0.5', 'nap', 'm_pow']]],
            # [0, 6, [['soma', '0.5', 'nap', 'h_pow']]],
            # [0, 6, [['soma', '0.5', 'kdr', 'n_pow']]],
            # [0, 6, [['soma', '0.5', 'hcn_slow', 'n_pow']]],

            [-90, -30, [['soma', '0.5', 'nat', 'm_vh']]],
            [-90, -30, [['soma', '0.5', 'nat', 'h_vh']]],
            [-90, -30, [['soma', '0.5', 'nap', 'm_vh']]],
            [-90, -30, [['soma', '0.5', 'nap', 'h_vh']]],
            [-90, -30, [['soma', '0.5', 'kdr', 'n_vh']]],
            [-90, -30, [['soma', '0.5', 'hcn_slow', 'n_vh']]],

            [10, 25, [['soma', '0.5', 'nat', 'm_vs']]],
            [-25, -10, [['soma', '0.5', 'nat', 'h_vs']]],
            [10, 25, [['soma', '0.5', 'nap', 'm_vs']]],
            [-25, -10, [['soma', '0.5', 'nap', 'h_vs']]],
            [10, 25, [['soma', '0.5', 'kdr', 'n_vs']]],
            [-25, -10, [['soma', '0.5', 'hcn_slow', 'n_vs']]],

            [0, 1, [['soma', '0.5', 'nat', 'm_tau_min']]],
            [0, 1, [['soma', '0.5', 'nat', 'h_tau_min']]],
            [0, 1, [['soma', '0.5', 'nap', 'm_tau_min']]],
            [0, 1, [['soma', '0.5', 'nap', 'h_tau_min']]],
            [0, 1, [['soma', '0.5', 'kdr', 'n_tau_min']]],
            [0, 10, [['soma', '0.5', 'hcn_slow', 'n_tau_min']]],

            [5, 30, [['soma', '0.5', 'nat', 'm_tau_max']]],
            [5, 30, [['soma', '0.5', 'nat', 'h_tau_max']]],
            [0, 1, [['soma', '0.5', 'nap', 'm_tau_max']]],
            [0, 30, [['soma', '0.5', 'nap', 'h_tau_max']]],
            [5, 30, [['soma', '0.5', 'kdr', 'n_tau_max']]],
            [100, 150, [['soma', '0.5', 'hcn_slow', 'n_tau_max']]],

            [0, 1, [['soma', '0.5', 'nat', 'm_tau_delta']]],
            [0, 1, [['soma', '0.5', 'nat', 'h_tau_delta']]],
            [0, 1, [['soma', '0.5', 'nap', 'm_tau_delta']]],
            [0, 1, [['soma', '0.5', 'nap', 'h_tau_delta']]],
            [0, 1, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
            [0, 1, [['soma', '0.5', 'hcn_slow', 'n_tau_delta']]],
            ]
lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
n_params = len(variable_keys)

# load models and get parameter values
load_mechanism_dir(mechanism_dir)
params = np.zeros((len(model_dirs), n_params))
for i, model_dir in enumerate(model_dirs):
    model_dir = os.path.join(model_dir, 'cell.json')
    cell = Cell.from_modeldir(model_dir)
    params[i, :] = [cell.get_attr(keys[0]) for keys in variable_keys]

# plot
cmap = matplotlib.cm.get_cmap('gnuplot')
colors = [cmap(x) for x in np.linspace(0, 1, len(model_dirs))]

# pl.figure()
# for i, c in enumerate(colors):
#     pl.plot(i, 0, color=c, marker='o', label='model '+str(i+1))
# pl.legend(fontsize=16)
# pl.show()

n_params_half = int(np.ceil(n_params / 2))
fig, ax = pl.subplots(2, n_params_half)
for i, m in enumerate(model_dirs):
    idx = 0
    for j, p in enumerate(variable_keys):
        if j >= n_params_half:
            idx = 1
        #ax[idx, j%n_params_half].set_xlabel(p[0][-2]+'\n'+p[0][-1])
        ax[idx, j % n_params_half].plot(0, params[i, j], color=colors[i], marker='o')
        ax[idx, j % n_params_half].set_xticks([0])
        ax[idx, j % n_params_half].set_xticklabels([p[0][-2] + '\n' + p[0][-1]])
        ax[idx, j % n_params_half].set_ylim(lower_bounds[j], upper_bounds[j])
        for label in ax[idx, j % n_params_half].get_xticklabels():
            label.set_rotation(90)
        # print p, params[i, j]
        assert lower_bounds[j] <= params[i, j] <= upper_bounds[j]
pl.subplots_adjust(wspace=5, left=0.05, right=0.95, top=0.95, bottom=0.15, hspace=0.35)
#pl.tight_layout()
pl.show()
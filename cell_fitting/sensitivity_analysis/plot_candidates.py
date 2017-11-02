import os
import numpy as np
from cell_fitting.optimization.helpers import get_lowerbound_upperbound_keys
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as pl
pl.style.use('paper')


# save dir
save_dir_data = '/home/cf/Phd/server/cns/server/results/sensitivity_analysis/'
save_dir_analysis = os.path.join('../results/sensitivity_analysis/', 'analysis_2017-10-10_new')
save_dir_plots = os.path.join(save_dir_analysis, 'plots', 'distributions')

if not os.path.exists(save_dir_plots):
    os.makedirs(save_dir_plots)

return_characteristics = np.load(os.path.join(save_dir_analysis, 'return_characteristics.npy'))
characteristics_mat = np.load(os.path.join(save_dir_analysis, 'characteristics_mat.npy'))
candidate_mat = np.load(os.path.join(save_dir_analysis, 'candidate_mat.npy'))
candidate_idxs = np.load(os.path.join(save_dir_analysis, 'candidate_idxs.npy'))  # TODO

# select candidate by characteritics
range_candidates = [
    [50, 150, 'AP_amp'],
    [0.1, 2.0, 'AP_width'],
    [0, 40, 'fAHP_amp'],
    [0, 40, 'DAP_amp'],
    [0, 20, 'DAP_deflection'],
    [0, 70, 'DAP_width'],
    [0, 20, 'DAP_time'],
    [-np.inf, np.inf, 'DAP_lin_slope'],
    [-np.inf, np.inf, 'DAP_exp_slope']
]
lower_bounds, upper_bounds, characteristics = get_lowerbound_upperbound_keys(range_candidates)

candidates_in_range = []
for candidate_idx, candidate_characteristics in zip(candidate_idxs, characteristics_mat):
    if np.all(lower_bounds < candidate_characteristics) and np.all(candidate_characteristics < upper_bounds):
        candidates_in_range.append(candidate_idx)

for candidate_idx in candidates_in_range:
    v = np.load(os.path.join(save_dir_data, candidate_idx[0], candidate_idx[1], 'v.npy'))
    t = np.load(os.path.join(save_dir_data, candidate_idx[0], 't.npy'))

    pl.figure()
    pl.plot(t, v)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential')
    pl.show()
import os
import matplotlib.pyplot as pl
import numpy as np
from cell_fitting.optimization.helpers import get_lowerbound_upperbound_keys
from cell_fitting.sensitivity_analysis import update_cell

pl.style.use('paper')


# save dir
save_dir_data = '/home/cf/Phd/server/cns/server/results/sensitivity_analysis/'
save_dir_analysis = os.path.join('../results/sensitivity_analysis/', 'analysis_2017-10-10')
save_dir_plots = os.path.join(save_dir_analysis, 'plots', 'distributions')

if not os.path.exists(save_dir_plots):
    os.makedirs(save_dir_plots)

return_characteristics = np.load(os.path.join(save_dir_analysis, 'return_characteristics.npy'))
characteristics_mat = np.load(os.path.join(save_dir_analysis, 'characteristics_mat.npy'))
candidate_mat = np.load(os.path.join(save_dir_analysis, 'candidate_mat.npy'))
candidate_idxs = np.load(os.path.join(save_dir_analysis, 'candidate_idxs.npy'))

# select candidate by characteristics
range_candidates = [
    [50, 150, 'AP_amp'],
    [0.1, 2.0, 'AP_width'],
    [0, 40, 'fAHP_amp'],
    [0, 40, 'DAP_amp'],
    [3, 20, 'DAP_deflection'],
    [0, 70, 'DAP_width'],
    [0, 6, 'DAP_time'],
    [-np.inf, np.inf, 'DAP_lin_slope'],
    [-np.inf, np.inf, 'DAP_exp_slope']
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
lower_bounds, upper_bounds, characteristics = get_lowerbound_upperbound_keys(range_candidates)

candidates_in_range = []
for candidate_idx, candidate, candidate_characteristics in zip(candidate_idxs, candidate_mat, characteristics_mat):
    if np.all(lower_bounds < candidate_characteristics) and np.all(candidate_characteristics < upper_bounds):
        candidates_in_range.append((candidate_idx, candidate))

x = 0
for candidate_idx, candidate in candidates_in_range:
    print candidate_idx
    v = np.load(os.path.join(save_dir_data, candidate_idx[0], candidate_idx[1], 'v.npy'))
    t = np.load(os.path.join(save_dir_data, candidate_idx[0], 't.npy'))

    pl.figure()
    pl.plot(t, v)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential')
    pl.tight_layout()
    pl.show()

    # save as cell
    from nrn_wrapper import Cell, load_mechanism_dir
    import json

    print candidate
    with open(os.path.join(save_dir_analysis, 'params.json'), 'r') as f:
        params = json.load(f)
    model_dir = params['model_dir']
    mechanism_dir = params['mechanism_dir']
    if x == 0:
        load_mechanism_dir(mechanism_dir)
        x = 1
    cell = Cell.from_modeldir(model_dir)
    _, _, variable_keys = get_lowerbound_upperbound_keys(params['variables'])
    cell.insert_mechanisms(variable_keys)
    update_cell(cell, candidate, variable_keys)
    for k, v in zip(variable_keys, candidate):
        cell.update_attr(k[0], v)
    with open(os.path.join(save_dir_data, candidate_idx[0], candidate_idx[1], 'cell.json'), 'w') as f:
        json.dump(cell.get_dict(), f, indent=4)


    # 2017-10-10_14:00:01/67982;
    # 2017-10-10_14:00:01/8945; 2017-10-10_14:00:02/15636; 2017-10-10_14:00:02/69043; 2017-10-10_14:00:01/2058
    # 2017-10-10_14:00:01/15123;

import matplotlib.pyplot as pl
import numpy as np
import os
import pandas as pd
from cell_fitting.new_optimization.evaluation.doubleramp_current_threshold import get_current_threshold, plot_current_threshold
from cell_characteristics import to_idx
pl.style.use('paper')


save_dir = './plots'
data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
PP_params_dir = '/home/cf/Phd/DAP-Project/cell_data/PP_params2.csv'
protocol = 'PP'
v_rest_shift = -16
protocol_idx = 0

#cells = ['2015_08_05b', '2015_08_05c', '2015_08_06d', '2015_08_10a', '2015_08_11d',
#         '2015_08_11e', '2015_08_11f']  # 05b, 11f only 10 amp increase, 05c, 06d 20
cell_id = '2014_07_02a'
run_idx = 3

step_amps = [-0.1, 0, 0.1]
AP_threshold = 0
PP_params = pd.read_csv(PP_params_dir)
PP_params['cell_id'].ffill(inplace=True)
PP_params_cell = PP_params[PP_params['cell_id'] == cell_id].iloc[run_idx]
len_ramp3_times = PP_params_cell['len_ramp3_times']
delta_first = PP_params_cell['delta_first']
delta_ramp = PP_params_cell['delta_ramp']
start_ramp2_idx = PP_params_cell['start_ramp2_idx']  # TODO: 48700 for 2015
ramp3_times = np.arange(delta_first, len_ramp3_times * delta_ramp + delta_ramp, delta_ramp)

save_dir_cell = os.path.join(save_dir, 'PP', cell_id, str(run_idx))
current_thresholds = [0] * len(step_amps)

# get amps
for i, step_amp in enumerate(step_amps):
    step_str = 'step_%.1f(nA)' % step_amp

    v_mat = np.load(os.path.join(save_dir_cell, step_str, 'v_mat.npy'))
    start_amp = PP_params_cell['ramp3_amp']
    ramp3_amps = start_amp + np.arange(len(v_mat[:, 0, 0])) * 0.05

    current_thresholds[i] = get_current_threshold(v_mat, ramp3_amps, ramp3_times, start_ramp2_idx, AP_threshold)

t = np.load(os.path.join(save_dir_cell, step_str, 't.npy'))
dt = t[1] - t[0]
print dt
start_ramp1 = to_idx(20, dt)
v_dap = v_mat[0, 0, start_ramp1:start_ramp1+to_idx(ramp3_times[-1]+ramp3_times[0], dt)]
t_dap = np.arange(len(v_dap)) * dt

plot_current_threshold(current_thresholds, ramp3_times, step_amps, ramp3_amps[0], ramp3_amps[-1], v_dap, t_dap,
                       save_dir_cell, legend_loc='upper left')  #'center right')


# TODO: plot current threshold with y as percentage of ramp_amp or threshold in rampIV
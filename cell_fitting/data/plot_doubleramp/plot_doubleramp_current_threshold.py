import os
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from cell_characteristics import to_idx
from cell_fitting.optimization.evaluation.plot_double_ramp.doubleramp_current_threshold import get_current_threshold, \
    plot_current_threshold, save_diff_current_threshold, plot_current_threshold_compare_in_vivo
pl.style.use('paper')


save_dir = '../plots'
data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
PP_params_dir = '/home/cf/Phd/DAP-Project/cell_data/PP_params2.csv'
protocol = 'PP'
v_rest_shift = -16
protocol_idx = 0

#cells = ['2015_08_05b', '2015_08_05c', '2015_08_06d', '2015_08_10a', '2015_08_11d',
#         '2015_08_11e', '2015_08_11f']  # 05b, 11f only 10 amp increase, 05c, 06d 20
cells = ['2014_07_10b', '2014_07_03a', '2014_07_08d', '2014_07_09c', '2014_07_09e', '2014_07_09f', '2014_07_10d']
run_idxs = [3, 7, 4, 5, 8, 1, 5]
cell_id = cells[0]
run_idx = run_idxs[0]

# load current threshold
current_threshold_rampIV = float(np.loadtxt(os.path.join(save_dir, 'rampIV', 'rat', cell_id, 'current_threshold.txt')))


step_amps = [-0.1, 0, 0.1]
AP_threshold = -20
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
    t = np.load(os.path.join(save_dir_cell, step_str, 't.npy'))
    dt = t[1] - t[0]
    start_amp = PP_params_cell['ramp3_amp']
    ramp3_amps = start_amp + np.arange(len(v_mat[:, 0, 0])) * 0.05

    current_thresholds[i] = get_current_threshold(v_mat, ramp3_amps, ramp3_times, start_ramp2_idx, dt, AP_threshold)

print dt
start_ramp1 = to_idx(20, dt)
v_dap = v_mat[0, 0, start_ramp1:start_ramp1+to_idx(ramp3_times[-1]+ramp3_times[0]+10, dt)]
t_dap = np.arange(len(v_dap)) * dt

# save difference of current threshold at DAP max and from rampIV
save_diff_current_threshold(current_threshold_rampIV, current_thresholds,
                            os.path.join(save_dir, 'PP', cell_id, 'diff_current_threshold.txt'))

plot_current_threshold(current_thresholds, current_threshold_rampIV, ramp3_times, step_amps, ramp3_amps[0],
                       ramp3_amps[-1], v_dap, t_dap, save_dir_cell, legend_loc='upper right')  #'center right')
plot_current_threshold_compare_in_vivo(current_thresholds, current_threshold_rampIV, ramp3_times, step_amps, ramp3_amps[0],
                       ramp3_amps[-1], v_dap, t_dap, save_dir_cell, legend_loc='upper right')
pl.show()
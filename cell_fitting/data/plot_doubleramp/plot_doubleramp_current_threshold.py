import os
import json
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
protocol_idx = 0

#cell_ids = ['2015_08_05c', '2015_08_06d', '2015_08_10a', '2015_08_11e', '2015_08_11f']  # 05b, 11f only 10 amp increase, 05c, 06d 20
cell_ids = ['2014_07_10b', '2014_07_03a', '2014_07_08d', '2014_07_09c', '2014_07_09e', '2014_07_09f', '2014_07_10d']
run_idxs = [3, 7, 4, 5, 8, 1, 5]
cell_ids = ['2015_08_06d']
run_idxs = [0]
cell_id = cell_ids[0]
run_idx = run_idxs[0]

# load current threshold
current_threshold_rampIV = float(np.loadtxt(os.path.join(save_dir, 'rampIV', 'rat', cell_id, 'current_threshold.txt')))

step_amps = [-0.1, 0, 0.1]
step_idx_dict = {-0.1: 0, 0: 1, 0.1: 2}
AP_threshold = -20
PP_params = pd.read_csv(PP_params_dir)
PP_params['cell_id'].ffill(inplace=True)
PP_params_cell = PP_params[PP_params['cell_id'] == cell_id].iloc[run_idx]
len_ramp3_times = PP_params_cell['len_ramp3_times']
delta_first = PP_params_cell['delta_first']
delta_ramp = PP_params_cell['delta_ramp']
start_ramp2_idx = int(PP_params_cell['start_ramp2_idx'])  # TODO: 48700 for 2015
ramp3_times = np.arange(delta_first, len_ramp3_times * delta_ramp + delta_ramp, delta_ramp)

save_dir_cell = os.path.join(save_dir, 'PP', cell_id, str(run_idx))
current_thresholds = [0] * len(step_amps)

# get amps
v_mat = np.load(os.path.join(save_dir_cell, 'v_mat.npy'))
for step_amp in step_amps:
    step_str = 'step_%.1f(nA)' % step_amp
    t = np.load(os.path.join(save_dir_cell, step_str, 't.npy'))
    dt = t[1] - t[0]
    start_amp = PP_params_cell['ramp3_amp']
    ramp3_amps = start_amp + np.arange(len(v_mat[:, 0, 0, step_idx_dict[step_amp]])) * 0.05

    current_thresholds[step_idx_dict[step_amp]] = get_current_threshold(v_mat[:, :, :, step_idx_dict[step_amp]],
                                                                        ramp3_amps, ramp3_times, start_ramp2_idx, dt,
                                                                        AP_threshold)

print dt
start_ramp1 = to_idx(20, dt)
v_dap = v_mat[0, 0, start_ramp1:start_ramp1+to_idx(ramp3_times[-1]+ramp3_times[0]+10, dt), 1]
t_dap = np.arange(len(v_dap)) * dt

# save
current_thresholds_lists = [list(c) for c in current_thresholds]
current_threshold_dict = dict(current_thresholds=current_thresholds_lists,
                              current_threshold_rampIV=[current_threshold_rampIV],
                              ramp3_times=list(ramp3_times), step_amps=list(step_amps), ramp3_amps=list(ramp3_amps),
                              v_dap=list(v_dap), t_dap=list(t_dap))
with open(os.path.join(save_dir, 'PP', cell_id, 'current_threshold_dict.json'), 'w') as f:
    json.dump(current_threshold_dict, f)

# save difference of current threshold at DAP max and from rampIV
save_diff_current_threshold(current_threshold_rampIV, current_thresholds,
                            os.path.join(save_dir, 'PP', cell_id, 'diff_current_threshold.txt'))
np.savetxt(os.path.join(save_dir, 'PP', cell_id, 'current_threshold_DAP.txt'),
           np.array([np.nanmin(current_thresholds[1])]))
np.savetxt(os.path.join(save_dir, 'PP', cell_id, 'current_threshold_rest.txt'),
           np.array([current_threshold_rampIV]))

plot_current_threshold(current_thresholds, current_threshold_rampIV, ramp3_times, step_amps, ramp3_amps[0],
                       ramp3_amps[-1], v_dap, t_dap, save_dir_cell, legend_loc='upper right')  #'center right')
plot_current_threshold_compare_in_vivo(current_thresholds, current_threshold_rampIV, ramp3_times, step_amps, ramp3_amps[0],
                       ramp3_amps[-1], v_dap, t_dap, save_dir_cell, legend_loc='upper right')
pl.show()
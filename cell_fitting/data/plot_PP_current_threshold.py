import matplotlib.pyplot as pl
import numpy as np
import os
import pandas as pd
from cell_fitting.read_heka import get_protocols_same_base
from cell_fitting.new_optimization.evaluation.doubleramp_current_threshold import get_current_threshold, plot_current_threshold
from cell_characteristics.analyze_APs import to_idx
pl.style.use('paper')


save_dir = './plots'
data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
data_dir_PP_params = '/home/cf/Phd/DAP-Project/cell_data/PP_params.csv'
protocol = 'PP'
v_rest_shift = -16
protocol_idx = 0

cells = ['2015_08_05b', '2015_08_05c', '2015_08_06d', '2015_08_10a', '2015_08_11d',
         '2015_08_11e', '2015_08_11f']  # 05b, 11f only 10 amp increase, 05c, 06d 20
step_amps = [-0.1, 0, 0.1]
AP_threshold = 0
delta_ramp = 2
delta_first = 3
ramp3_times = np.arange(delta_first, 10 * delta_ramp + delta_ramp, delta_ramp)
start_ramp2 = 48700
PP_params = pd.read_csv(data_dir_PP_params)
PP_params.cell.ffill(inplace=True)

for c_idx, cell in enumerate(cells):
    print cell
    save_dir_cell = os.path.join(save_dir, 'PP', cell)
    current_thresholds = [0] * len(step_amps)

    # get amps
    PP_params_cell = PP_params[PP_params.cell == cell]

    for i, step_amp in enumerate(step_amps):
        step_str = 'step_%.1f(nA)' % step_amp

        v_mat = np.load(os.path.join(save_dir_cell, step_str, 'v_mat.npy'))
        if cell == '2015_08_11e':
            start_amp = PP_params_cell['ramp3_amp'].iloc[1]
        else:
            start_amp = PP_params_cell['ramp3_amp'].iloc[0]
        ramp3_amps = start_amp + np.arange(len(v_mat[:, 0, 0])) * 0.05

        current_thresholds[i] = get_current_threshold(v_mat, ramp3_amps, ramp3_times, start_ramp2, AP_threshold)

    t = np.load(os.path.join(save_dir_cell, step_str, 't.npy'))
    dt = t[1] - t[0]
    start_ramp1 = to_idx(20, dt)
    v_dap = v_mat[0, 0, start_ramp1:start_ramp1+to_idx(ramp3_times[-1]+ramp3_times[0], dt)]
    t_dap = np.arange(len(v_dap)) * dt

    plot_current_threshold(current_thresholds, ramp3_times, step_amps, ramp3_amps[0], ramp3_amps[-1], v_dap, t_dap,
                           save_dir_cell, legend_loc='upper left')  #'center right')


# TODO: plot current threshold with y as percentage of ramp_amp or threshold in rampIV
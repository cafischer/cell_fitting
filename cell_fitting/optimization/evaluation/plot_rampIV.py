import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import os
from cell_fitting.optimization.fitter import iclamp_handling_onset
from nrn_wrapper import Cell
import time
from cell_fitting.data import shift_v_rest
from cell_fitting.read_heka import get_v_and_t_from_heka, get_sweep_index_for_amp, get_i_inj_from_function
from cell_characteristics import to_idx
from cell_characteristics.analyze_APs import get_AP_onset_idxs
pl.style.use('paper')

__author__ = 'caro'


def simulate_rampIV(cell, ramp_amp, v_init=-75):
    protocol = 'rampIV'
    dt = 0.01
    tstop = 161.99  # ms
    sweep_idx = get_sweep_index_for_amp(ramp_amp, protocol)
    i_inj = get_i_inj_from_function(protocol, [sweep_idx], tstop, dt)[0]

    # get simulation parameters
    simulation_params = {'sec': ('soma', None), 'i_inj': i_inj, 'v_init': v_init, 'tstop': tstop,
                         'dt': dt, 'celsius': 35, 'onset': 200}

    # record v
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)
    return v, t, i_inj


def find_current_threshold(cell):
    for ramp_amp in np.arange(0.1, 4.0+0.1, 0.1):
        v, t, i_inj = simulate_rampIV(cell, ramp_amp)
        start = np.where(i_inj)[0][0]
        onset_idxs = get_AP_onset_idxs(v[start:], threshold=0)
        if len(onset_idxs) >= 1:
            return ramp_amp
    return None


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/6'
    #save_dir = '/home/cf/Phd/server/cns/server/results/sensitivity_analysis/2017-10-10_14:00:01/3519'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'
    ramp_amp = 3.1
    #data_dir = '../../data/2015_08_26b/rampIV/'+str(ramp_amp)+'(nA).csv'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data/2015_08_26b.dat'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    start_time = time.time()
    v, t, _ = simulate_rampIV(cell, ramp_amp, v_init=-75)
    end_time = time.time()
    print 'Runtime (sec): ', end_time - start_time

    # current to elicit AP
    current_threshold = find_current_threshold(cell)
    print 'Current threshold: %.2f nA' % current_threshold

    #data = pd.read_csv(data_dir)
    v_data, t_data = get_v_and_t_from_heka(data_dir, 'rampIV', sweep_idxs=[get_sweep_index_for_amp(ramp_amp, 'rampIV')])
    v_data = shift_v_rest(v_data[0], -16)
    t_data = t_data[0]
    dt = t_data[1] - t_data[0]

    # rmse
    rmse = np.sqrt(np.mean((v - v_data)**2))
    dap_start = to_idx(13.5, dt)
    dap_end = to_idx(80, dt)
    rmse_dap = np.sqrt(np.mean((v[dap_start:dap_end] - v_data[dap_start:dap_end]) ** 2))
    rmse_dap = np.sqrt(np.mean((v[dap_start:dap_end] - v_data[dap_start:dap_end]) ** 2))
    print 'RMSE: %.2f mV' % rmse
    print 'RMSE DAP: %.2f mV' % rmse_dap

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'rampIV')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    np.savetxt(os.path.join(save_dir_img, 'current_threshold.txt'), np.array([current_threshold]))

    pl.figure()
    #pl.title(str(np.round(ramp_amp, 2)) + ' nA')
    pl.plot(t_data, v_data, 'k', label='Exp. Data')
    pl.plot(t, v, 'r', label='Model')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential (mV)')
    pl.legend(loc='upper right')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'rampIV_with_data' + str(np.round(ramp_amp, 2)) + 'nA' + '.png'))
    pl.show()

    pl.figure()
    pl.plot(t, v, 'r', label='Model')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential (mV)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'rampIV' + str(np.round(ramp_amp, 2)) + 'nA' + '.png'))
    pl.show()
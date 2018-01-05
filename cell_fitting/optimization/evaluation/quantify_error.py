import os

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from cell_characteristics.analyze_APs import get_AP_max_idx, get_fAHP_min_idx_using_splines, get_DAP_max_idx
from nrn_wrapper import Cell

from cell_fitting.optimization.errfuns import rms
from cell_fitting.optimization.evaluation.rampIV import simulate_rampIV

pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir = '../../results/server/2017-07-27_09:18:59/22/L-BFGS-B/'
    model_dir = os.path.join(save_dir, 'model', 'cell.json')
    #save_dir = '../../results/hand_tuning/cell_2017-07-24_13:59:54_21_0/'
    #model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'
    ramp_amp = 3.0
    data_dir = '../../data/2015_08_26b/vrest-75/simulate_rampIV/'+str(ramp_amp)+'(nA).csv'

    AP_threshold = -30

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # simulate
    v, t = simulate_rampIV(cell, ramp_amp, v_init=-75)
    data = pd.read_csv(data_dir)
    v_data = data.v.values
    t_data = data.t.values

    # find DAP
    dt = t_data[1] - t_data[0]
    AP_end = len(v_data)
    AP_max_idx = get_AP_max_idx(v_data, 0, AP_end)
    std = np.std(v_data[0:int(round(10/dt))])
    w = np.ones(len(v_data)) / std
    fAHP_min_idx = get_fAHP_min_idx_using_splines(v_data, t_data, AP_max_idx, AP_end, w=w, interval=int(round(5/dt)))
    DAP_max = get_DAP_max_idx(v_data, fAHP_min_idx, AP_end, dist_to_max=int(round(20/dt)))
    sAHP_min_idx = get_fAHP_min_idx_using_splines(v_data, t_data, DAP_max, AP_end, w=w)

    # print
    print 'RMS whole trace: %.2f' % rms(v, data.v)
    print 'RMS DAP: %.2f' % rms(v[fAHP_min_idx:sAHP_min_idx], v_data[fAHP_min_idx: sAHP_min_idx])

    # plot
    pl.figure()
    pl.plot(data.t, data.v, 'k', label='Exp. Data')
    pl.plot(t, v, 'r', label='Model')
    pl.axvline(data.t[fAHP_min_idx], color='0.5')
    pl.axvline(data.t[sAHP_min_idx], color='0.5')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend()
    pl.tight_layout()
    pl.show()
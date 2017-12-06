import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import os
from cell_fitting.optimization.fitter import iclamp_handling_onset
from nrn_wrapper import Cell
import time
from cell_fitting.read_heka import get_i_inj_from_function, get_sweep_index_for_amp
from cell_fitting.read_heka import get_v_and_t_from_heka, get_sweep_index_for_amp
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


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/6'
    #save_dir = '/home/cf/Phd/server/cns/server/results/sensitivity_analysis/2017-10-10_14:00:01/3519'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'
    ramp_amp = 3.0
    #data_dir = '../../data/2015_08_26b/rampIV/'+str(ramp_amp)+'(nA).csv'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data/2015_08_26b.dat'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    start_time = time.time()
    v, t, _ = simulate_rampIV(cell, ramp_amp, v_init=-75)
    end_time = time.time()
    print 'Runtime (sec): ', end_time - start_time

    #data = pd.read_csv(data_dir)
    v_data, t_data = get_v_and_t_from_heka(data_dir, 'rampIV', sweep_idxs=[get_sweep_index_for_amp(3.1, 'rampIV')])

    # plot
    save_img = os.path.join(save_dir, 'img', 'rampIV')
    if not os.path.exists(save_img):
        os.makedirs(save_img)

    pl.figure()
    #pl.title(str(np.round(ramp_amp, 2)) + ' nA')
    pl.plot(t_data[0], v_data[0]-8, 'k', label='Exp. Data')
    pl.plot(t, v, 'r', label='Model')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    #pl.legend(loc='upper right')
    pl.tight_layout()
    #pl.savefig(os.path.join(save_img, 'rampIV' + str(np.round(ramp_amp, 2)) + 'nA'+'.png'))
    pl.show()
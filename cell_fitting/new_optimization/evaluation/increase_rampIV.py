import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import os
from cell_fitting.new_optimization.fitter import iclamp_handling_onset
from nrn_wrapper import Cell
pl.style.use('paper')

__author__ = 'caro'


def get_ramp(start_idx, peak_idx, end_idx, amp_before, ramp_amp, amp_after):
    diff_idx = end_idx - start_idx
    half_diff_up = peak_idx - start_idx + 1
    half_diff_down = end_idx - peak_idx - 1
    i_exp = np.zeros(diff_idx)
    i_exp[:half_diff_up] = np.linspace(amp_before, ramp_amp, half_diff_up)
    i_exp[half_diff_up:] = np.linspace(ramp_amp, amp_after, half_diff_down+1)[1:]
    return i_exp


def rampIV(cell, ramp_amp, v_init=-75):

    dt = 0.01
    ramp_st_ms = 10  # ms
    ramp_peak_ms = 10.8  # ms
    ramp_end_ms = 12  # ms
    tstop = 161.99  # ms

    ramp_st = int(round(ramp_st_ms / dt))
    ramp_peak = int(round(ramp_peak_ms / dt))
    ramp_end = int(round(ramp_end_ms / dt)) + 1

    t_exp = np.arange(0, tstop + dt, dt)
    i_exp = np.zeros(len(t_exp))
    i_exp[ramp_st:ramp_end] = get_ramp(ramp_st, ramp_peak, ramp_end, 0, ramp_amp, 0)

    # get simulation parameters
    simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': v_init, 'tstop': t_exp[-1],
                         'dt': dt, 'celsius': 35, 'onset': 200}

    # record v
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    return v, t


if __name__ == '__main__':
    # parameters
    #save_dir = '../../results/server/2017-08-16_09:41:34/148/L-BFGS-B/'
    #model_dir = os.path.join(save_dir, 'model', 'cell.json')
    save_dir = '../../results/hand_tuning/test0/'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'
    ramp_amp = 3.0
    data_dir = '../../data/2015_08_26b/vrest-75/rampIV/'+str(ramp_amp)+'(nA).csv'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    v, t = rampIV(cell, ramp_amp, v_init=-75)

    data = pd.read_csv(data_dir)

    # plot
    save_img = os.path.join(save_dir, 'img', 'rampIV')
    if not os.path.exists(save_img):
        os.makedirs(save_img)

    pl.figure()
    #pl.title(str(np.round(ramp_amp, 2)) + ' nA')
    pl.plot(data.t, data.v, 'k', label='Exp. Data')
    pl.plot(t, v, 'r', label='Model')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend(loc='upper right')
    pl.tight_layout()
    pl.savefig(os.path.join(save_img, 'rampIV' + str(np.round(ramp_amp, 2)) + 'nA'+'.png'))
    pl.show()
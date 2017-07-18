import pylab as pl
import numpy as np
import os
from new_optimization.fitter import iclamp_handling_onset
from nrn_wrapper import Cell

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

    # test i_exp
    #pl.figure()
    #pl.plot(t, i_exp)
    #pl.show()

    # plot
    pl.figure()
    pl.plot(t, v, 'r', label=str(np.round(ramp_amp, 2)) + ' nA')
    pl.xlabel('Time $(ms)$', fontsize=16)
    pl.ylabel('Membrane potential $(mV)$', fontsize=16)
    pl.legend(loc='upper right', fontsize=16)
    pl.show()


if __name__ == '__main__':
    # parameters
    data_dir = '../../data/2015_08_26b/vrest-75/rampIV/3.0(nA).csv'
    #save_dir = '../../results/server/2017-07-06_13:50:52/434/L-BFGS-B/'
    #model_dir = os.path.join(save_dir, 'model', 'best_cell.json')
    save_dir = '../../results/hand_tuning/cell434_1/'
    model_dir = '../../results/hand_tuning/cell434_1/cell.json'
    mechanism_dir = '../../model/channels/vavoulis'
    ramp_amp = 3.0

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    rampIV(cell, ramp_amp, v_init=-75)
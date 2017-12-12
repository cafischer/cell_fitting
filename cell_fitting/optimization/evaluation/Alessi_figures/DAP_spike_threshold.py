from __future__ import division

import os

import matplotlib.pyplot as pl
import numpy as np
from cell_characteristics import to_idx
from nrn_wrapper import Cell

from cell_fitting.optimization.evaluation.Alessi_figures import find_AP_current
from cell_fitting.optimization.evaluation.plot_IV import get_step
from cell_fitting.optimization.simulate import iclamp_handling_onset

pl.style.use('paper')

__author__ = 'caro'


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/1'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    save_dir_hold = os.path.join(save_dir, 'img', 'DAP_at_different_holding_potentials')

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # load holding potentials and amps
    hold_potentials = np.load(os.path.join(save_dir_hold, 'hold_potentials.npy'))
    hold_amps = np.load(os.path.join(save_dir_hold, 'hold_amps.npy'))
    hold_potential = np.nan  # hold_potentials[-1]  #
    hold_amp = 0  # hold_amps[-1]  #
    v_init = hold_potential if not np.isnan(hold_potential) else -75
    save_dir_img = os.path.join(save_dir, 'img', 'DAP_spike_threshold', 'hold_potential' + str(hold_potential))

    # simulation params
    dt = 0.001
    first_step_st_ms = 50  # ms
    first_step_end_ms = first_step_st_ms + 1  # ms
    second_step_st_ms = first_step_st_ms + 15  # ms
    second_step_end_ms = second_step_st_ms + 1  # ms
    tstop = 240  # ms
    test_step_amps = np.arange(0, 3.0, 0.1)
    AP_threshold = -20
    celsius = 35
    onset = 200

    # find right AP current 1st spike
    i_hold = get_step(0, to_idx(tstop, dt) + 1, to_idx(tstop, dt) + 1, hold_amp)
    first_spike_amp = find_AP_current(cell, i_hold, test_step_amps, first_step_st_ms,
                                      first_step_end_ms, AP_threshold, v_init, tstop, dt, onset=onset, celsius=celsius,
                                      plot=False)

    # find right AP current 2nd spike
    i_hold = get_step(0, to_idx(tstop, dt) + 1, to_idx(tstop, dt) + 1, hold_amp)
    i_step = get_step(to_idx(first_step_st_ms, dt), to_idx(first_step_end_ms, dt), to_idx(tstop, dt) + 1,
                          first_spike_amp)
    second_spike_amp = find_AP_current(cell, i_hold + i_step, test_step_amps, second_step_st_ms,
                                      second_step_end_ms, AP_threshold, v_init, tstop, dt, onset=onset, celsius=celsius,
                                      plot=False)

    # simulate
    i_hold = get_step(0, to_idx(tstop, dt) + 1, to_idx(tstop, dt) + 1, hold_amp)
    i_step = get_step(to_idx(first_step_st_ms, dt), to_idx(first_step_end_ms, dt), to_idx(tstop, dt) + 1,
                          first_spike_amp)
    i_step2 = get_step(to_idx(second_step_st_ms, dt), to_idx(second_step_end_ms, dt), to_idx(tstop, dt) + 1,
                          second_spike_amp)
    i_exp = i_hold + i_step + i_step2
    simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': v_init,
                         'tstop': tstop, 'dt': dt, 'celsius': celsius, 'onset': onset}
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    # plot
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    np.save(os.path.join(save_dir_img, 'first_spike_amp.npy'), first_spike_amp)
    np.save(os.path.join(save_dir_img, 'second_spike_amp.npy'), second_spike_amp)

    pl.figure()
    pl.plot(t, v, color='r', label='Model')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'v.png'))
    #pl.show()

    pl.figure()
    pl.bar(0, first_spike_amp, color='r')
    pl.bar(1, second_spike_amp, color='r')
    pl.xticks([0, 1], ['1st Pulse', '2nd Pulse'])
    pl.xlabel('Time (ms)')
    pl.ylabel('Current (nA)')
    pl.ylim(test_step_amps[0], test_step_amps[-1])
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'amps.png'))
    pl.show()
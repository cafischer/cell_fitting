from __future__ import division
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.optimization.evaluation.Alessi_figures import find_hold_amps, find_AP_current, \
    simulate_with_step_and_holding_current
from cell_fitting.optimization.evaluation.plot_IV import get_step
from cell_characteristics import to_idx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
pl.style.use('paper')

__author__ = 'caro'


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/1'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    save_dir_img = os.path.join(save_dir, 'img', 'DAP_at_different_holding_potentials')
    amps_given = False

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # different holding potentials
    hold_potentials = [-87, -82, -72, -67]
    dt = 0.001
    step_st_ms = 50  # ms
    step_end_ms = step_st_ms + 1  # ms
    tstop = 240  # ms
    tstop_hold = 100
    test_hold_amps = np.arange(-2, 1, 0.001)
    test_step_amps = np.arange(2.0, 8.0, 0.1)
    AP_threshold = -20

    # compute/load holding and spike amplitudes
    if amps_given:
        step_amp_spike = np.load(os.path.join(save_dir_img, 'step_amp_spike.npy'))
        hold_potentials = np.load(os.path.join(save_dir_img, 'hold_potentials.npy'))
        hold_amps = np.load(os.path.join(save_dir_img, 'hold_amps.npy'))
    else:
        hold_amps = find_hold_amps(cell, hold_potentials, test_hold_amps, tstop_hold, dt)
        print 'Holding amplitudes: ', hold_amps

        i_hold = get_step(0, to_idx(tstop, dt) + 1, to_idx(tstop, dt) + 1, hold_amps[0])
        step_amp_spike = find_AP_current(cell, i_hold, hold_amps[0], test_step_amps, step_st_ms, step_end_ms,
                                         AP_threshold, hold_potentials[0], tstop, dt)
        print 'Spike amplitude: ', step_amp_spike

    # simulate different holding potentials with step current
    v_mat, t = simulate_with_step_and_holding_current(cell, hold_potentials, hold_amps, step_amp_spike, step_st_ms,
                                                      step_end_ms, tstop, dt, onset=200, celsius=35)

    # save and plot
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    np.save(os.path.join(save_dir_img, 'v_mat.npy'), v_mat)
    np.save(os.path.join(save_dir_img, 't.npy'), t)
    np.save(os.path.join(save_dir_img, 'hold_potentials.npy'), hold_potentials)
    np.save(os.path.join(save_dir_img, 'hold_amps.npy'), hold_amps)
    np.save(os.path.join(save_dir_img, 'step_amp_spike.npy'), step_amp_spike)

    cmap = pl.cm.get_cmap('Reds')
    colors = [cmap(x) for x in np.linspace(0.2, 1.0, len(v_mat))]
    pl.figure()
    for i, v in enumerate(v_mat):
        pl.plot(t, v, color=colors[i], label='Model' if i == 0 else None)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'v.png'))
    pl.show()
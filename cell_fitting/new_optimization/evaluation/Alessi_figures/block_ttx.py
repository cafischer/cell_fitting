from __future__ import division
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.new_optimization.evaluation.Alessi_figures import find_hold_amps, \
    simulate_with_step_and_holding_current
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
pl.style.use('paper')

__author__ = 'caro'


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/5'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    percent_block = 0.2
    use_same_hold_amps = False
    if use_same_hold_amps:
        save_dir_img = os.path.join(save_dir, 'img', 'TTX', 'same_amps', 'percent_block' + str(percent_block))
    else:
        save_dir_img = os.path.join(save_dir, 'img', 'TTX', 'new_amps', 'percent_block'+str(percent_block))
    save_dir_hold = os.path.join(save_dir, 'img', 'DAP_at_different_holding_potentials')

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # load holding potentials and amplitudes
    dt = 0.001
    step_st_ms = 50  # ms
    step_end_ms = step_st_ms + 1  # ms
    tstop = 240  # ms
    test_hold_amps = np.arange(-2, 2, 0.001)
    step_amp_spike = np.load(os.path.join(save_dir_hold, 'step_amp_spike.npy'))
    hold_potentials = np.load(os.path.join(save_dir_hold, 'hold_potentials.npy'))
    hold_amps = np.load(os.path.join(save_dir_hold, 'hold_amps.npy'))
    v_mat = np.load(os.path.join(save_dir_hold, 'v_mat.npy'))

    # TTX application
    cell.soma(.5).nat.gbar = cell.soma(.5).nat.gbar * (1 - percent_block)
    cell.soma(.5).nap.gbar = cell.soma(.5).nap.gbar * (1 - percent_block)

    # find hold_amps
    if use_same_hold_amps:
        hold_amps_block = hold_amps
    else:
        hold_amps_block = find_hold_amps(cell, hold_potentials, test_hold_amps, tstop, dt, plot=False)
        print 'Holding amplitudes: ' + str(hold_amps_block)

    # simulate different holding potentials with step current
    v_mat_block, t = simulate_with_step_and_holding_current(cell, hold_potentials, hold_amps_block, step_amp_spike,
                                                            step_st_ms, step_end_ms, tstop, dt, onset=200, celsius=35)

    # plot
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    np.save(os.path.join(save_dir_img, 'v_mat.npy'), v_mat)
    np.save(os.path.join(save_dir_img, 'v_mat_block.npy'), v_mat_block)
    np.save(os.path.join(save_dir_img, 't.npy'), t)
    np.save(os.path.join(save_dir_img, 'hold_potentials.npy'), hold_potentials)
    np.save(os.path.join(save_dir_img, 'hold_amps.npy'), hold_amps)
    np.save(os.path.join(save_dir_img, 'hold_amps_block.npy'), hold_amps_block)
    np.save(os.path.join(save_dir_img, 'step_amp_spike.npy'), step_amp_spike)

    for i, hold_potential in enumerate(hold_potentials):
        pl.figure()
        pl.plot(t, v_mat[i], 'r', label='without block')
        pl.plot(t, v_mat_block[i], 'o', label='with ' + str(percent_block*100) + '% block of Na$^+$')
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane potential (mV)')
        pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'v_'+str(hold_potential)+'.png'))
        pl.show()
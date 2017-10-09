from __future__ import division
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.new_optimization.evaluation.IV import get_step, get_IV
from cell_fitting.optimization.simulate import iclamp_handling_onset, simulate_currents
from cell_characteristics.analyze_APs import get_AP_onset_idxs
import matplotlib.pyplot as pl

pl.style.use('paper')

__author__ = 'caro'

if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    save_img = os.path.join(save_dir, 'img', 'DAP_at_different_holding_potentials')

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # different holding potentials
    dt = 0.001
    step_st_ms = 50  # ms
    step_end_ms = step_st_ms + 1  # ms
    tstop = 240  # ms
    tstop_hold = 100
    test_hold_amps = np.arange(-2, 1, 0.001)
    step_amps = np.arange(2.0, 8.0, 0.1)
    AP_threshold = -20

    # load membrane potentials
    v_mat = np.load(os.path.join(save_img, 'v_mat.npy'))
    t = np.load(os.path.join(save_img, 't.npy'))
    hold_potentials = np.load(os.path.join(save_img, 'hold_potentials.npy'))
    hold_potentials = np.insert(hold_potentials, 2, -75)
    hold_amps = np.load(os.path.join(save_img, 'hold_amps.npy'))
    hold_amps = np.insert(hold_amps, 2, 0)
    step_amp_spike = np.load(os.path.join(save_img, 'step_amp_spike.npy'))

    # simulate different holding potentials with step current
    vs = []
    currents = []
    for i, hold_amp in enumerate(hold_amps):
        i_step = get_step(int(round(step_st_ms / dt)), int(round(step_end_ms / dt)), int(round(tstop / dt)) + 1,
                          step_amp_spike)
        i_hold = get_step(0, int(round(tstop / dt)) + 1, int(round(tstop / dt)) + 1, hold_amp)
        i_exp = i_hold + i_step

        simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': hold_potentials[i],
                             'tstop': tstop, 'dt': dt, 'celsius': 35, 'onset': 200}
        v, t, _ = iclamp_handling_onset(cell, **simulation_params)

        # extra: simulate currents
        current, channel_list = simulate_currents(cell, simulation_params, False)

        vs.append(v)
        currents.append(current)

    # plot
    save_img = os.path.join(save_dir, 'img', 'DAP_at_different_holding_potentials')
    if not os.path.exists(save_img):
        os.makedirs(save_img)

    np.save(os.path.join(save_img, 'v_mat.npy'), vs)
    np.save(os.path.join(save_img, 't.npy'), t)
    #np.save(os.path.join(save_img, 'hold_potentials.npy'), hold_potentials)
    #np.save(os.path.join(save_img, 'hold_amps.npy'), hold_amps)

    cmap = pl.cm.get_cmap('Reds')
    colors = cmap(np.arange(len(currents[0])) / len(currents[0]))
    colors_h = cmap(np.arange(len(hold_potentials)) / len(hold_potentials))
    for j in range(len(currents[0])):
        pl.figure()
        pl.title(channel_list[j])
        for i, hold_potential in enumerate(hold_potentials):
            pl.plot(t, currents[i][j], color=colors_h[i], label=str(hold_potential))
        #pl.plot(t, v, color=colors[i], label='Model' if i == 0 else None)
        pl.xlabel('Time (ms)')
        pl.ylabel('Current (mA/$cm^2$)')
        pl.legend()
        pl.tight_layout()
        pl.savefig(os.path.join(save_img, 'v.png'))
        pl.show()
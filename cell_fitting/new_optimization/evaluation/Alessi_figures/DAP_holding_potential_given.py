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
    #save_dir = '../../../results/server/2017-07-27_09:18:59/22/L-BFGS-B'
    #model_dir = os.path.join(save_dir, 'model', 'cell.json')
    #save_dir = '../../../results/hand_tuning/test0'
    #model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    save_img = os.path.join(save_dir, 'img', 'DAP_at_different_holding_potentials')

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # load holding potentials and amplitudes
    dt = 0.001
    step_st_ms = 50  # ms
    step_end_ms = step_st_ms + 1  # ms
    tstop = 240  # ms
    step_amp_spike = np.load(os.path.join(save_img, 'step_amp_spike.npy'))
    hold_potentials = np.load(os.path.join(save_img, 'hold_potentials.npy'))
    hold_amps = np.load(os.path.join(save_img, 'hold_amps.npy'))

    # simulate different holding potentials with step current
    vs = []
    for i, hold_amp in enumerate(hold_amps):
        i_step = get_step(int(round(step_st_ms/dt)), int(round(step_end_ms/dt)), int(round(tstop/dt))+1, step_amp_spike)
        i_hold = get_step(0, int(round(tstop/dt)) + 1, int(round(tstop/dt))+1, hold_amp)
        i_exp = i_hold + i_step

        simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': hold_potentials[i],
                             'tstop': tstop, 'dt': dt, 'celsius': 35, 'onset': 200}
        v, t, _ = iclamp_handling_onset(cell, **simulation_params)

        # extra: simulate currents
        currents = simulate_currents(cell, simulation_params, True)

        vs.append(v)

    # plot
    save_img = os.path.join(save_dir, 'img', 'DAP_at_different_holding_potentials')
    if not os.path.exists(save_img):
        os.makedirs(save_img)

    np.save(os.path.join(save_img, 'v_mat.npy'), vs)
    np.save(os.path.join(save_img, 't.npy'), t)
    np.save(os.path.join(save_img, 'hold_potentials.npy'), hold_potentials)
    np.save(os.path.join(save_img, 'hold_amps.npy'), hold_amps)
    np.save(os.path.join(save_img, 'step_amp_spike.npy'), step_amp_spike)

    cmap = pl.cm.get_cmap('Reds')
    colors = [cmap(x) for x in np.linspace(0.2, 1.0, len(vs))]
    pl.figure()
    for i, v in enumerate(vs):
        pl.plot(t, v, color=colors[i], label='Model' if i == 0 else None)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_img, 'v.png'))
    pl.show()
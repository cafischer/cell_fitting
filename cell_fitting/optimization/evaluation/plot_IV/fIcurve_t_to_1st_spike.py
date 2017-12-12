from __future__ import division
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.optimization.evaluation.fI_curve import get_step, get_IV, compute_time_to_1st_spike
import matplotlib
import matplotlib.pyplot as pl
pl.style.use('paper')


__author__ = 'caro'


if __name__ == '__main__':
    # parameters
    #save_dir = '../../../results/server/2017-07-27_09:18:59/22/L-BFGS-B'
    #model_dir = os.path.join(save_dir, 'model', 'cell.json')
    save_dir = '../../../results/hand_tuning/test0'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # high resolution FI-curve
    step_st_ms = 250  # ms
    tstop = 1100  # ms
    step_end_ms = tstop - step_st_ms  # ms
    step_amps = np.arange(0.1, 0.4, 0.01)
    vs = []
    i_injs = []
    for step_amp in step_amps:
        v, t, i_inj = get_IV(cell, step_amp, get_step, step_st_ms, step_end_ms, tstop, v_init=-75, dt=0.001)
        vs.append(v)
        i_injs.append((i_inj))

    vs = np.vstack(vs)
    i_injs = np.vstack(i_injs)
    amps, firing_rates = compute_time_to_1st_spike(vs, i_injs, t)

    # plot
    save_img = os.path.join(save_dir, 'img', 'plot_IV', 'time_to_1st_spike')
    if not os.path.exists(save_img):
        os.makedirs(save_img)

    cmap = matplotlib.cm.get_cmap('Reds')
    colors = [cmap(x) for x in np.linspace(0.2, 1.0, len(vs))]
    pl.figure()
    for i, v in enumerate(vs):
        pl.plot(t, v, color=colors[i], label='Model' if i == 0 else None)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_img, 'plot_IV.png'))

    pl.figure()
    for i, v in enumerate(vs):
        pl.plot(t, v, color=colors[i], label='Model' if i == 0 else None)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.xlim(250, 350)
    pl.ylim(-75, -55)
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_img, 'IV_zoom_1st_AP.png'))

    pl.figure()
    pl.plot(amps, firing_rates, '-or')
    pl.ylabel('Time to 1st AP (ms)')
    pl.xlabel('Current (nA)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_img, 'time_to_1st_spike.png'))
    pl.show()

from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from nrn_wrapper import Cell
from cell_characteristics.fIcurve import compute_fIcurve
from cell_fitting.optimization.evaluation.plot_IV import get_IV, get_step
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/5'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # high resolution FI-curve
    step_st_ms = 200  # ms
    tstop = 1200  # ms
    step_end_ms = tstop - step_st_ms  # ms
    step_amps = np.arange(-2.5, -2.0, 0.05)
    print step_amps
    vs = []
    i_injs = []
    for step_amp in step_amps:
        v, t, i_inj = get_IV(cell, step_amp, get_step, step_st_ms, step_end_ms, tstop, v_init=-75, dt=0.01)
        vs.append(v)
        i_injs.append(i_inj)

    vs = np.vstack(vs)
    i_injs = np.vstack(i_injs)
    amps, firing_rates = compute_fIcurve(vs, i_injs, t)

    # plot
    save_img = os.path.join(save_dir, 'img', 'plot_IV', 'high_res', 'super_high')
    if not os.path.exists(save_img):
        os.makedirs(save_img)

    colors = pl.cm.get_cmap('Reds')(np.linspace(0.2, 1.0, len(vs)))
    pl.figure()
    for i, v in enumerate(vs):
        pl.plot(t, v, color=colors[i], label='Model' if i == 0 else None)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.legend()
    pl.tight_layout()
    #pl.savefig(os.path.join(save_img, 'plot_IV.png'))

    pl.figure()
    for i, v in enumerate(vs):
        pl.plot(t, v, color=colors[i], label='Model' if i == 0 else None)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.xlim(100, 800)
    pl.ylim(-75, -55)
    pl.legend()
    pl.tight_layout()
    #pl.savefig(os.path.join(save_img, 'IV_zoom1.png'))

    pl.figure()
    for i, v in enumerate(vs):
        pl.plot(t, v, color=colors[i], label='Model' if i == 0 else None)
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.xlim(100, 300)
    pl.ylim(-75, -55)
    pl.legend()
    pl.tight_layout()
    #pl.savefig(os.path.join(save_img, 'IV_zoom2.png'))

    pl.figure()
    pl.plot(amps, firing_rates, '-or')
    pl.ylabel('Firing rate (APs/ms)')
    pl.xlabel('Current (nA)')
    pl.tight_layout()
    #pl.savefig(os.path.join(save_img, 'fI_curve.png'))
    pl.show()

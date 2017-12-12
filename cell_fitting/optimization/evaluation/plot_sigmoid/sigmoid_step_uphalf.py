from __future__ import division

import os

import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell

from cell_fitting.optimization.evaluation.plot_sigmoid import sig_upper_half
from cell_fitting.optimization.fitter import iclamp_handling_onset
from cell_fitting.optimization.simulate import simulate_currents

pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/1'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # simulate
    dt = 0.01
    offset_dur = 100
    len_sig = 100
    a_range = np.arange(0.05, 0.16, 0.05)
    # 6: np.arange(0.05, 0.16, 0.05) switch no doublet -> doublet;
    # 5: np.arange(0.05, 0.16, 0.05) always slow doublet; 4: no doublet;
    # 3: np.arange(0.05, 0.16, 0.05) always slow doublet; 2: np.arange(0.05, 0.16, 0.05) always slow doublet;
    # 1: np.arange(0.05, 0.16, 0.05) no doublet
    x = np.arange(-len_sig, len_sig, dt)
    tstop = len(x) * dt + offset_dur

    v_traces = []
    t_traces = []
    i_inj_traces = []
    currents = []
    for a in a_range:
        i_inj = sig_upper_half(0, 1, a, x)
        simulation_params = {'sec': ('soma', None), 'i_inj': i_inj, 'v_init': -75, 'tstop': tstop,
                             'dt': dt, 'celsius': 35, 'onset': 200}

        v, t, i_inj = iclamp_handling_onset(cell, **simulation_params)
        v_traces.append(v)
        t_traces.append(t)
        i_inj_traces.append(i_inj)

        current, channel_list = simulate_currents(cell, simulation_params)
        currents.append(current)


    # plot
    save_img = os.path.join(save_dir, 'img', 'plot_IV', 'sigmoid_step', 'upper_half')
    if not os.path.exists(save_img):
        os.makedirs(save_img)

    fig, ax = pl.subplots(3, 1, sharex=True)
    for i, (v, t) in enumerate(zip(v_traces, t_traces)):
        ax[i].plot(t, v, 'k', label='Slope: '+str(a_range[i]))
        pl.xlabel('Time (ms)')
        pl.xlim(50, 150)
        leg = ax[i].legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=14)
        for item in leg.legendHandles:
            item.set_visible(False)
    fig.text(0.02, 0.55, 'Membrane Potential (mV)', va='center', rotation='vertical', fontsize=18)
    pl.subplots_adjust(bottom=0.14, left=0.15, top=0.96, right=0.96)
    pl.savefig(os.path.join(save_img, 'v.png'))
    pl.show()

    # pl.figure()
    # for i_inj, t in zip(i_inj_traces, t_traces):
    #     pl.plot(t[:len(i_inj)], i_inj)
    # pl.xlabel('Time (ms)')
    # pl.ylabel('Current (nA)')
    # pl.xlim(90, 160)
    # pl.tight_layout()
    # pl.savefig(os.path.join(save_img, 'i_inj.png'))
    # #pl.show()
    #
    # pl.figure(figsize=(13, 8))
    # cmap = pl.cm.get_cmap()
    # colors = cmap(np.linspace(0, 1, len(channel_list)))
    # for i, current in enumerate(currents):
    #     for c, c_n, color in zip(current, channel_list, colors):
    #         pl.plot(t_traces[0], -c, color=color, alpha=(i+1)/len(currents), label=c_n if i == 1 else None)
    # pl.xlabel('Time (ms)')
    # pl.ylabel('Current (mA/cm$^2$)')
    # pl.xlim(90, 160)
    # pl.ylim(-0.25, 0.25)
    # pl.legend()
    # pl.tight_layout()
    # pl.savefig(os.path.join(save_img, 'currents.png'))
    # pl.show()

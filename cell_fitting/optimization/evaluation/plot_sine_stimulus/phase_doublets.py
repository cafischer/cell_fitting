from __future__ import division
import matplotlib.pyplot as pl
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.optimization.evaluation.plot_sine_stimulus import simulate_sine_stimulus
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_fitting.util import init_nan
from cell_characteristics import to_idx
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/3'
    #save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/server_17_12_04/2018-01-05_14:13:33/154/L-BFGS-B'
    #save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/server_17_12_04/2017-12-26_08:14:12/185/L-BFGS-B'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'

    # model 3: 0.6, 0.3 oder 0.6, 0.4  -> shorter doublets are rather outside
    # model 3: 0.4, 0.4  -> doublets get shorter closer to the inside
    # model 5: 0.5, 0.2 -> doublets get shorter closer to the inside
    # model 185: 0.8, 0.3 -> doublets get shorter closer to the inside

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # parameters
    max_ISI = 15  # ms
    AP_threshold = -10
    freq1 = 0.25
    freq2 = 5
    amp1 = 0.6  # 0.5
    amp2 = 0.3  # 0.2
    sine1_dur = 1. / freq1 / 2 * 1000
    onset_dur = 500
    offset_dur = 500
    dt = 0.01

    save_dir_img = os.path.join(save_dir, 'img', 'sine_stimulus', 'doublets', 'phase')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    v, t, _ = simulate_sine_stimulus(cell, amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt)
    v = v[to_idx(onset_dur, dt): -to_idx(offset_dur, dt)]
    t = t[to_idx(onset_dur, dt): -to_idx(offset_dur, dt)] - t[to_idx(onset_dur, dt)]
    onsets = get_AP_onset_idxs(v, AP_threshold)
    phases_sine_slow = (np.linspace(0, 180, len(t)))  # half a sine
    phase_APs = phases_sine_slow[onsets]
    ISIs = np.diff(onsets * dt)
    short_ISIs_idx = ISIs < max_ISI
    short_ISIs = ISIs[short_ISIs_idx]
    phase_short_ISIs = phase_APs[:-1][short_ISIs_idx]

    pl.figure(figsize=(18, 8))
    pl.plot(t, v, 'k')
    pl.plot(t[onsets[:-1][short_ISIs_idx]], v[onsets[:-1][short_ISIs_idx]], 'or')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane potential (mV)')
    pl.savefig(os.path.join(save_dir_img, 'v.png'))
    #pl.show()

    # plot
    fig, ax = pl.subplots()
    ax.plot(short_ISIs, phase_short_ISIs, 'ok')
    ax.set_xlabel('ISI (ms)')
    ax.set_ylabel('Phase')
    ax.set_yticks(np.arange(0, 181, 10))
    ax.set_yticklabels((np.arange(0, 181, 10) + 270) % 360)  # peak shall be at 360
    ax.set_xlim(0, 15)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'short_ISI_phases.png'))
    #pl.show()
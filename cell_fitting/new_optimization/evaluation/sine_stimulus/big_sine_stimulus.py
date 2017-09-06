from __future__ import division
import matplotlib.pyplot as pl
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.new_optimization.evaluation.sine_stimulus import apply_sine_stimulus
from cell_characteristics.analyze_APs import get_AP_onset_idxs
pl.style.use('paper')


def plot_two_colored_rectangle(ax, center, w, h, colors=('r', 'b')):
    center1 = (center[0] - w, center[1] - h/2)
    center2 = (center[0], center[1] - h/2)
    ax.add_patch(Rectangle(center1, w, h, color=colors[0]))
    ax.add_patch(Rectangle(center2, w, h, color=colors[1]))


def divide0(x, y):
    if y == 0:
        return 0
    return x / y


if __name__ == '__main__':
    # parameters
    save_dir = '../../../results/server/2017-07-27_09:18:59/22/L-BFGS-B/'
    model_dir = os.path.join(save_dir, 'model', 'cell.json')
    #save_dir = '../../results/hand_tuning/cell_2017-07-24_13:59:54_21_0'
    #model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # parameters
    AP_threshold = -10
    amp2 = 0  # no modulating sine
    freq2 = 0
    onset_dur = 500
    offset_dur = 500
    dt = 0.01

    d_amp = 0.1
    amp1s = np.arange(0.1, 1.0+d_amp, d_amp)
    d_sine = 500
    sine1_durs = np.arange(500, 2000+d_sine, d_sine)
    n_APs_up = np.zeros((len(amp1s), len(sine1_durs)))
    n_APs_down = np.zeros((len(amp1s), len(sine1_durs)))

    for i, amp1 in enumerate(amp1s):
        for j, sine1_dur in enumerate(sine1_durs):
            v, t, _ = apply_sine_stimulus(cell, amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt)
            onsets = get_AP_onset_idxs(v, AP_threshold)
            n_APs_up[i, j] = len(onsets[onsets < len(t)/2])
            n_APs_down[i, j] = len(onsets[onsets > len(t)/2])

            # pl.figure()
            # pl.plot(t, v)
            # pl.show()

    max_APs = max(np.max(n_APs_up), np.max(n_APs_down))
    cmap = pl.get_cmap('Greys')

    fig, ax = pl.subplots()
    for i, amp1 in enumerate(amp1s):
        for j, sine1_dur in enumerate(sine1_durs):
            colors = (cmap(divide0(n_APs_up[i, j], max_APs)), cmap(divide0(n_APs_down[i, j], max_APs)))
            plot_two_colored_rectangle(ax, (amp1, sine1_dur), w=d_amp/2, h=d_sine/4, colors=colors)

    pl.xlim(amp1s[0]-d_amp, amp1s[-1]+d_amp)
    pl.ylim(sine1_durs[0]-d_sine, sine1_durs[-1]+d_sine)
    sm = pl.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=max_APs))
    sm.set_array(np.array([0, max_APs]))
    pl.colorbar(sm)
    pl.tight_layout()
    pl.show()
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
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # parameters
    AP_threshold = -10
    amp1 = 0.6
    sine1_dur = 2000
    freq1 = 1. / (2*sine1_dur/1000)
    freq2 = 5
    onset_dur = 500
    offset_dur = 500
    dt = 0.01

    d_amp = 0.1
    amp1s = np.arange(0.1, 1.0+d_amp, d_amp)
    amp2s = np.arange(0.1, 1.0+d_amp, d_amp)

    ISI_1st = init_nan((len(amp1s), len(amp2s)))

    save_dir_img = os.path.join(save_dir, 'img', 'sine_stimulus', 'when_doublet', 'start',
                                'freq2_'+str(freq2) + '_freq1_'+str(freq1))
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    for i, amp1 in enumerate(amp1s):
        for j, amp2 in enumerate(amp2s):
            v, t, _ = simulate_sine_stimulus(cell, amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt)
            onsets = get_AP_onset_idxs(v, AP_threshold)
            half_osc_idx = to_idx(1./freq2 / 2 * 1000, dt)
            onsets_1st_osc = onsets[np.logical_and(to_idx(onset_dur, dt) < onsets,
                                                   onsets < to_idx(onset_dur, dt) + half_osc_idx)]
            onsets_2nd_osc = onsets[np.logical_and(to_idx(onset_dur, dt) + half_osc_idx < onsets,
                                    onsets < to_idx(onset_dur, dt) + 3 * half_osc_idx)]
            if len(onsets_1st_osc) >= 2:
                ISI_1st[i, j] = (onsets_1st_osc[1] - onsets_1st_osc[0]) * dt
            elif len(onsets_2nd_osc) >= 2:
                ISI_1st[i, j] = (onsets_2nd_osc[1] - onsets_2nd_osc[0]) * dt
            else:
                ISI_1st[i, j] = np.nan
            print ISI_1st[i, j]

            pl.figure(figsize=(18, 8))
            pl.plot(t, v, 'k', linewidth=1.0)
            pl.xlabel('Time (ms)')
            pl.ylabel('Membrane Potential (mV)')
            pl.ylim(-95, 55)
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_img, 'v_amp1_%.2f_amp2_%.2f.png' % (amp1, amp2)))
            #pl.show()

    # plot
    cmap = pl.get_cmap('viridis')
    ISI_max = 15
    norm = Normalize(vmin=0, vmax=ISI_max)
    fig, ax = pl.subplots()
    for i, amp1 in enumerate(amp1s):
        for j, amp2 in enumerate(amp2s):
            if not np.isnan(ISI_1st[i, j]):
                if ISI_1st[i, j] > ISI_max:
                    w = d_amp / 2
                    ax.add_patch(Rectangle((amp1 - w / 2, amp2 - w / 2), w, w, color='r'))
                else:
                    c = cmap(norm(ISI_1st[i, j]))
                    w = d_amp/2
                    ax.add_patch(Rectangle((amp1-w/2, amp2-w/2), w, w, color=c))

    pl.xlim(amp1s[0]-d_amp/2, amp1s[-1]+d_amp/2)
    pl.ylim(amp2s[0]-d_amp/2, amp2s[-1]+d_amp/2)
    pl.xlabel('Amplitude Ramp (nA)')
    pl.ylabel('Amplitude Modulation (nA)')
    sm = pl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(np.array([0, ISI_max]))
    cb = pl.colorbar(sm)
    cb.ax.get_yaxis().labelpad = 20
    cb.ax.set_ylabel('$ ISI_{2nd-1st}$', rotation=270)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'ISI.png'))
    #pl.show()
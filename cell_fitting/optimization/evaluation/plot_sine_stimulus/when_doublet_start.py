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
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/1'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # parameters
    AP_threshold = -10
    amp1 = 0.6
    sine1_dur = 1000
    onset_dur = 500
    offset_dur = 500
    dt = 0.01

    d_amp = 0.1
    amp2s = np.arange(0.1, 1.0+d_amp, d_amp)
    d_freq = 2
    freq2s = np.arange(3, 15+d_freq, d_freq)

    ISI_first = init_nan((len(amp2s), len(freq2s)))

    save_dir_img = os.path.join(save_dir, 'img', 'plot_sine_stimulus', 'when_doublet', 'start', 'amp1_'+str(amp1)+'_dur1_'+str(sine1_dur))
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    for i, amp2 in enumerate(amp2s):
        for j, freq2 in enumerate(freq2s):
            v, t, _ = simulate_sine_stimulus(cell, amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt)
            onsets = get_AP_onset_idxs(v, AP_threshold)  # use only period in the middle
            if len(onsets) >= 2:
                if (onsets[1] - onsets[0]) * dt < 1/2 * 1/freq2 * 1000:
                    ISI_first[i, j] = (onsets[1] - onsets[0]) * dt
                    print ISI_first[i, j]

            pl.figure(figsize=(18, 8))
            pl.plot(t, v, 'k', linewidth=1.0)
            pl.xlabel('Time (ms)')
            pl.ylabel('Membrane Potential (mV)')
            pl.ylim(-95, 55)
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_img, 'v_'+str(amp2)+'_'+str(freq2)+'.png'))
            #pl.show()

    # plot
    cmap = pl.get_cmap('viridis')
    ISI_max = 15
    norm = Normalize(vmin=0, vmax=ISI_max)
    fig, ax = pl.subplots()
    for i, amp2 in enumerate(amp2s):
        for j, freq2 in enumerate(freq2s):
            if not np.isnan(ISI_first[i, j]):
                if ISI_first[i, j] > ISI_max:
                    w = d_amp / 2
                    h = d_freq / 6
                    ax.add_patch(Rectangle((amp2 - w / 2, freq2 - h / 2), w, h, color='r'))
                else:
                    c = cmap(norm(ISI_first[i, j]))
                    w = d_amp/2
                    h = d_freq/6
                    ax.add_patch(Rectangle((amp2-w/2, freq2-h/2), w, h, color=c))

    pl.xlim(amp2s[0]-d_amp/2, amp2s[-1]+d_amp/2)
    pl.ylim(freq2s[0]-d_freq/2, freq2s[-1]+d_freq/2)
    pl.xlabel('Amplitude (nA)')
    pl.ylabel('Frequency (Hz)')
    sm = pl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(np.array([0, ISI_max]))
    cb = pl.colorbar(sm)
    cb.ax.get_yaxis().labelpad = 20
    cb.ax.set_ylabel('$ ISI_{2nd-1st}$', rotation=270)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'ISI.png'))
    #pl.show()
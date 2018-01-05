from __future__ import division
import matplotlib.pyplot as pl
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.optimization.evaluation.plot_sine_stimulus import simulate_sine_stimulus
from cell_characteristics.analyze_APs import get_AP_onset_idxs
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
    amp2 = 0  # no modulating sine
    freq2 = 0
    onset_dur = 500
    offset_dur = 500
    dt = 0.01

    d_amp = 0.1
    amp1s = np.arange(0.1, 2.5+d_amp, d_amp)
    d_sine = 2000
    sine1_durs = np.arange(2000, 15000+d_sine, d_sine)
    #sine1_durs = np.array([1000, 5000, 10000])
    n_APs_up = np.zeros((len(amp1s), len(sine1_durs)))
    n_APs_down = np.zeros((len(amp1s), len(sine1_durs)))

    save_dir_img = os.path.join(save_dir, 'img', 'plot_sine_stimulus', 'bistability_ramp')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    for i, amp1 in enumerate(amp1s):
        for j, sine1_dur in enumerate(sine1_durs):
            v, t, _ = simulate_sine_stimulus(cell, amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt)
            onsets = get_AP_onset_idxs(v, AP_threshold)
            n_APs_up[i, j] = len(onsets[onsets < len(t)/2])
            n_APs_down[i, j] = len(onsets[onsets > len(t)/2])

            pl.figure()
            pl.plot(t, v, 'r')
            pl.xlabel('Time (ms)')
            pl.ylabel('Membrane potential (mV)')
            pl.savefig(os.path.join(save_dir_img, 'v_'+str(amp1)+'_'+str(sine1_dur)+'.png'))
            #pl.show()
            pl.close()

    # plot difference in spikes
    max_diff = np.max(np.abs(n_APs_up - n_APs_down))
    cmap = pl.get_cmap('coolwarm')
    norm = Normalize(vmin=-max_diff, vmax=max_diff)
    fig, ax = pl.subplots()
    for i, amp1 in enumerate(amp1s):
        for j, sine1_dur in enumerate(sine1_durs):
            if not (n_APs_up[i, j] == 0 and n_APs_down[i, j] == 0):
                c = cmap(norm(n_APs_up[i, j] - n_APs_down[i, j]))
                w = d_amp/2
                h = d_sine/6
                ax.add_patch(Rectangle((amp1-w/2, sine1_dur-h/2), w, h, color=c))

    pl.xlim(amp1s[0]-d_amp/2, amp1s[-1]+d_amp/2)
    pl.ylim(sine1_durs[0]-d_sine/2, sine1_durs[-1]+d_sine/2)
    pl.xlabel('Amplitude (nA)')
    pl.ylabel('Duration (ms)')
    sm = pl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(np.array([-max_diff, max_diff]))
    cb = pl.colorbar(sm)
    cb.ax.get_yaxis().labelpad = 20
    cb.ax.set_ylabel('$ \# APs_{first\ half} - \# APs_{second\ half}$', rotation=270)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'diff_AP_up_down.png'))
    #pl.show()
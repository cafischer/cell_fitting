from cell_characteristics.analyze_APs import get_AP_onset_idxs
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.optimization.evaluation.plot_sine_stimulus import simulate_sine_stimulus
import matplotlib.pyplot as pl


def get_instantaneous_firing_rate_and_doublets(v, t):
    short_ISI_lim = 10
    AP_threshold = -20
    dt = t[1] - t[0]

    onset_idxs = get_AP_onset_idxs(v, AP_threshold)
    ISIs = np.diff(onset_idxs) * dt
    f_inst = np.zeros(len(v))
    for i in range(len(onset_idxs)-1):
        f_inst[onset_idxs[i]:onset_idxs[i+1]] = 1./ISIs[i] * 1000  # to sec
    doublets = onset_idxs[np.where(ISIs < short_ISI_lim)] * dt

    return f_inst, doublets


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/6'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # apply stim
    amp1 = 0.3
    amp2 = 0.2
    #freq1 = 0.5  # 0.5: 1000, 0.25: 2000, 0.1: 5000, 0.05: 10000
    freq1s = [0.5, 0.25, 0.05]
    #freq2 = 5  # 1, 5, 20
    freq2s = [1, 5, 20]
    onset_dur = offset_dur = 500
    dt = 0.01

    ts = []
    i_injs = []
    f_insts = []
    doubletss = []
    freq2s_all = []
    freq1s_all = []
    for freq2 in freq2s:
        for freq1 in freq1s:
            sine_params = {'amp1': amp1, 'amp2': amp2, 'sine1_dur': 1./freq1 * 1000 / 2, 'freq2': freq2,
                           'onset_dur': onset_dur, 'offset_dur': offset_dur, 'dt': dt}

            # simulate
            v, t, i_inj = simulate_sine_stimulus(cell, amp1, amp2, 1./freq1 * 1000/2, freq2, onset_dur, offset_dur, dt)

            # instantaneous firing rate
            f_inst, doublets = get_instantaneous_firing_rate_and_doublets(v, t)

            ts.append(t)
            i_injs.append(i_inj)
            f_insts.append(f_inst)
            doubletss.append(doublets)
            freq1s_all.append(freq1)
            freq2s_all.append(freq2)

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'sine_stimulus', 'f_inst')  #+'_'+str(freq1)+'_'+str(freq2))
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    total_time = 20000

    fig, axes = pl.subplots(len(freq1s)*len(freq2s), 1, figsize=(21, 29.7))
    for i, ax in enumerate(axes):
        t = ts[i]
        ax.set_title('amp1: ' + str(amp1) + ', amp2: ' + str(amp2) +
                     ', freq1: ' + str(freq1s_all[i]) + ', freq2: ' + str(freq2s_all[i]),
                     fontsize=16)
        off = (total_time - t[-1]) / 2.
        ax2 = ax.twinx()
        ax2.plot(t+off, i_injs[i], 'orange')
        ax2.set_ylabel('Current (nA)', fontsize=16)
        ax2.spines['right'].set_visible(True)
        ax.vlines(doubletss[i], 0, 1, color='r')
        ax.plot(t+off, f_insts[i], 'b')
        ax.set_xlabel('Time (ms)', fontsize=16)
        ax.set_ylabel('Instantaneous \nFiring Rate (Hz)', fontsize=16)
        ax.set_xlim(0, total_time)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'amp1_'+str(amp1)+'_amp2_'+str(amp2)+'.pdf'))
    pl.show()
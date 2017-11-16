from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell
import os
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_fitting.optimization.evaluation.sine_stimulus import apply_sine_stimulus
from cell_characteristics import to_idx
pl.style.use('paper')


def plot_FI_with_color(param, param_name, I, F, I_step, F_step):
    norm = pl.Normalize(vmin=0, vmax=np.max(param))
    cmap = pl.cm.get_cmap('viridis')
    colors = cmap(norm(param))
    pl.figure()
    pl.scatter(I, F, color=colors, label='from sine')
    pl.plot(I_step, F_step, 'or', label='from step')
    pl.xlabel('Current (nA)')
    pl.ylabel('Firing rate (APs/ms)')
    sm = pl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(np.array([0, 1]))
    cbar = pl.colorbar(sm)
    cbar.set_label(param_name)
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'FI' + param_name + '.png'))
    #pl.show()


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/6'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'
    save_dir_fi_curve = os.path.join(save_dir, 'img', 'IV')

    AP_threshold = -10

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # apply stim
    amp1s = np.arange(0, 1.0, 0.2)
    amp2s = np.arange(0, 1.0, 0.2)
    sine1_durs = [1000, 2000, 5000, 10000]  # in Hz: 0.5, 0.25, 0.1, 0.05
    freq2s = np.arange(5, 25, 5)
    onset_dur = 500
    offset_dur = 500
    dt = 0.01

    F = []
    I = []
    Amp1s = []
    Amp2s = []
    Sine1_durs = []
    Freq2s = []
    for amp1 in amp1s:
        for amp2 in amp2s:
            for sine1_dur in sine1_durs:
                for freq2 in freq2s:
                    sine_params = {'amp1': amp1, 'amp2': amp2, 'sine1_dur': sine1_dur, 'freq2': freq2, 'onset_dur': onset_dur,
                                   'offset_dur': offset_dur, 'dt': dt}

                    v, t, i_inj = apply_sine_stimulus(cell, amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt)

                    start_idx = to_idx(onset_dur+sine1_dur/2-250, dt)
                    end_idx = to_idx(onset_dur+sine1_dur/2+250, dt)

                    # pl.figure()
                    # pl.plot(t[start_idx:end_idx], v[start_idx:end_idx])
                    # pl.show()

                    n_APs = len(get_AP_onset_idxs(v[start_idx:end_idx], AP_threshold))
                    delta_t = t[end_idx] - t[start_idx]
                    i_avg = np.mean(i_inj[start_idx: end_idx])
                    F.append(n_APs/delta_t)
                    I.append(i_avg)
                    Amp1s.append(amp1)
                    Amp2s.append(amp2)
                    Sine1_durs.append(sine1_dur)
                    Freq2s.append(freq2)

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'IV', 'from_sine', 'fixed_int')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    F_step = np.load(os.path.join(save_dir_fi_curve, 'firing_rates.npy'))
    I_step = np.load(os.path.join(save_dir_fi_curve, 'amps_greater0.npy'))

    pl.figure()
    pl.scatter(I, F, color='k', label='from sine')
    pl.scatter(I_step, F_step, color='r', label='from step')
    pl.xlabel('Current (nA)')
    pl.ylabel('Firing rate (APs/ms)')
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'FI.png'))
    #pl.show()

    # plot for each parameter in color
    plot_FI_with_color(Amp1s, 'amp1', I, F, I_step, F_step)
    plot_FI_with_color(Amp2s, 'amp2', I, F, I_step, F_step)
    plot_FI_with_color(Freq2s, 'freq2', I, F, I_step, F_step)
    plot_FI_with_color(Sine1_durs, 'sine1_dur', I, F, I_step, F_step)
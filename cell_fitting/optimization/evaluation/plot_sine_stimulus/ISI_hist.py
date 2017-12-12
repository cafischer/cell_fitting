from __future__ import division
import os
import numpy as np
from grid_cell_stimuli.ISI_hist import get_ISI_hist, get_ISI_hists_into_outof_field, plot_ISI_hist, \
    plot_ISI_hist_into_outof_field
from cell_characteristics.analyze_APs import get_AP_onset_idxs
import matplotlib.pyplot as pl
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.evaluation.plot_sine_stimulus import apply_sine_stimulus
from cell_characteristics import to_idx
pl.style.use('paper')


if __name__ == '__main__':
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/'
    model_ids = range(1, 7)
    mechanism_dir = '../../../model/channels/vavoulis'
    load_mechanism_dir(mechanism_dir)

    amp1 = 0.6  # 0.5
    amp2 = 0.2  # 0.2
    freq1 = 0.1  # 0.5: 1000, 0.25: 2000, 0.1: 5000, 0.05: 10000
    sine1_dur = 1./freq1 * 1000 / 2
    freq2 = 5  # 5  # 20
    onset_dur = offset_dur = 500
    dt = 0.01

    for model_id in model_ids:
        # load model
        model_dir = os.path.join(save_dir, str(model_id), 'cell.json')
        cell = Cell.from_modeldir(model_dir)

        save_dir_img = os.path.join(save_dir, str(model_id), 'img', 'sine_stimulus', 'traces',
                                    str(amp1)+'_'+str(amp2)+'_'+str(freq1)+'_'+str(freq2), 'ISI_hist')
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        # simulate
        v, t, i_inj = apply_sine_stimulus(cell, amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt)

        # parameter
        AP_threshold = -30
        order = int(round(20 / dt))
        dist_to_AP = int(round(250 / dt))
        speed = 0.040  # cm/ms
        bins = np.arange(0, 200, 2)

        # rebuild theta stim (sine2)
        x = np.arange(0, sine1_dur + dt, dt)
        theta = amp2 * np.sin(2 * np.pi * x * freq2 / 1000)
        onset = np.zeros(to_idx(onset_dur, dt))
        offset = np.zeros(to_idx(offset_dur, dt))
        theta = np.concatenate((onset, theta, offset))

        # ISI histogram
        ISI_hist, ISIs = get_ISI_hist(v, t, AP_threshold, bins)

        # in and out field ISIs
        position = t * speed
        track_len = position[-1]
        field_pos = [track_len / 2]
        field_pos_idxs = [np.argmin(np.abs(position - p)) for p in field_pos]
        ISI_hist_into, ISI_hist_outof = get_ISI_hists_into_outof_field(v, t, AP_threshold, bins, field_pos_idxs)

        # save and plots
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        n_ISI = len(ISIs)
        n_doublets = np.sum(ISI_hist[bins[:-1] < 20])
        percent_doublets = n_doublets / n_ISI
        print('Percent doublets: ', percent_doublets)

        n_theta = np.sum(ISI_hist[np.logical_and(90 <= bins[:-1], bins[:-1] < 200)])
        percent_theta = n_theta / n_ISI
        print('Percent theta: ', percent_theta)

        plot_ISI_hist(ISI_hist, bins, save_dir_img)
        plot_ISI_hist_into_outof_field(ISI_hist_into, ISI_hist_outof, bins, save_dir_img)

        # spikes per up-period of theta stim
        period = to_idx(1/freq2*1000, dt)
        start_period = 0
        period_half = to_idx(period, 2)
        period_fourth = int(round(period / 4))
        onset_idx = to_idx(onset_dur, dt)
        offset_idx = to_idx(onset_dur, dt)
        period_starts = range(len(t))[onset_idx - period_fourth:-offset_idx:period]
        period_ends = range(len(t))[onset_idx + period_half + period_fourth:-offset_idx:period]
        period_starts = period_starts[:len(period_ends)]

        # pl.figure()
        # pl.plot(t, theta)
        # pl.plot(t[period_starts], theta[period_starts], 'og')
        # pl.plot(t[period_ends], theta[period_ends], 'or')
        # pl.show()

        AP_onsets = get_AP_onset_idxs(v, AP_threshold)
        n_APs_per_period = np.zeros(len(period_starts))
        for i, (s, e) in enumerate(zip(period_starts, period_ends)):
            n_APs_per_period[i] = np.sum(np.logical_and(s < AP_onsets, AP_onsets < e))

        pl.figure()
        pl.plot(range(1, len(period_starts) + 1), n_APs_per_period, 'ok')
        pl.ylabel('Count APs')
        pl.xlabel('Number of Period')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'n_APs_per_period.png'))
        #pl.show()

        # plot periods one under another
        colors = pl.cm.get_cmap('Reds')(np.linspace(0.2, 1.0, len(period_starts)))
        pl.figure()
        for i, (s, e) in enumerate(zip(period_starts, period_ends)):
            pl.plot(t[:e-s], v[s:e] + i * -10.0, c=colors[i], label=i, linewidth=1)
        pl.yticks([])
        pl.xlabel('Time (ms)')
        pl.ylabel('Membrane Potential (mV)')
        pl.xlim(0, 200)
        pl.ylim(-325, -45)
        pl.legend(fontsize=6, title='Period')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'periods.png'))
        #pl.show()

        # see how many spikes on each side
        n_periods = len(period_starts) if len(period_starts) % 2 == 0 else len(period_starts) - 1
        half_periods_idx = int(n_periods/2)
        n_AP_in_out_str = '# APs into: %i \n' % np.sum(n_APs_per_period[:half_periods_idx]) \
                          + '# APs out of: %i' % np.sum(n_APs_per_period[-half_periods_idx:])
        print n_AP_in_out_str
        with open(os.path.join(save_dir, 'n_APs_in_out.txt'), 'w') as f:
            f.write(n_AP_in_out_str)
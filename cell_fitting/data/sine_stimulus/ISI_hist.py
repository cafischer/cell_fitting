from __future__ import division
import os
import numpy as np
import json
from grid_cell_stimuli.ISI_hist import get_ISI_hist, get_ISI_hists_into_outof_field, plot_ISI_hist, \
    plot_ISI_hist_into_outof_field
from cell_characteristics.analyze_APs import get_AP_onset_idxs
import matplotlib.pyplot as pl
pl.style.use('paper')


if __name__ == '__main__':
    dur1 = 5000
    freq2 = 5
    save_dir = os.path.join('../plots/sine_stimulus', str(dur1)+'_'+str(freq2))
    cells = [os.path.split(d)[-1] for d in os.listdir(save_dir)]

    for cell in cells:
        save_dir_cell = os.path.join(save_dir, cell)

        # load
        v = np.load(os.path.join(save_dir_cell, 'v.npy'))
        t = np.load(os.path.join(save_dir_cell, 't.npy'))
        dt = t[1] - t[0]
        with open(os.path.join(save_dir_cell, 'sine_params.json'), 'r') as f:
            sine_params = json.load(f)

        # parameter
        AP_threshold = -30
        order = int(round(20 / dt))
        dist_to_AP = int(round(250 / dt))
        speed = 0.040  # cm/ms
        bins = np.arange(0, 200, 2)

        # rebuild theta stim (sine2)
        x = np.arange(0, sine_params['sine1_dur'] + sine_params['dt'], sine_params['dt'])
        theta = sine_params['amp2'] * np.sin(2 * np.pi * x * sine_params['freq2']/1000)
        onset = np.zeros(int(round(sine_params['onset_dur']/sine_params['dt'])))
        offset = np.zeros(int(round(sine_params['offset_dur']/sine_params['dt'])))
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
        save_dir_img = os.path.join(save_dir_cell, 'ISI_hist')
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

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
        period = int(round(1/sine_params['freq2']*1000/dt))
        start_period = 0
        period_half = int(round(period / 2))
        period_fourth = int(round(period / 4))
        onset_idx = int(round(sine_params['onset_dur']/sine_params['dt']))
        offset_idx = int(round(sine_params['onset_dur'] / sine_params['dt']))
        period_starts = range(len(t))[onset_idx - period_fourth:-offset_idx:period]
        period_ends = range(len(t))[onset_idx + period_half + period_fourth:-offset_idx:period]
        period_starts = period_starts[:len(period_ends)]

        # pl.figure()
        # pl.plot(t, theta)
        # pl.plot(t[ups_start], theta[ups_start], 'og')
        # pl.plot(t[ups_end], theta[ups_end], 'or')
        # pl.show()

        AP_onsets = get_AP_onset_idxs(v, AP_threshold)
        n_APs_per_period = np.zeros(len(period_starts))
        for i, (s, e) in enumerate(zip(period_starts, period_ends)):
            n_APs_per_period[i] = np.sum(np.logical_and(s < AP_onsets, AP_onsets < e))

        pl.figure()
        pl.plot(range(1, len(period_starts) + 1), n_APs_per_period, 'ok')
        pl.ylabel('Count APs')
        pl.xlabel('Number of Period')
        pl.savefig(os.path.join(save_dir_img, 'n_APs_per_up.png'))
        #pl.show()

        # plot periods one under another
        colors = pl.cm.get_cmap('Greys')(np.linspace(0.2, 1.0, len(period_starts)))
        pl.figure()
        for i, (s, e) in enumerate(zip(period_starts, period_ends)):
            pl.plot(t[:e - s], v[s:e] + i * -10.0, c=colors[i], label=i)
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
        half_periods_idx = int(n_periods / 2)
        n_AP_in_out_str = '# APs into: %i \n' % np.sum(n_APs_per_period[:half_periods_idx]) \
                          + '# APs out of: %i' % np.sum(n_APs_per_period[-half_periods_idx:])
        print n_AP_in_out_str
        with open(os.path.join(save_dir_img, 'n_APs_in_out.txt'), 'w') as f:
            f.write(n_AP_in_out_str)
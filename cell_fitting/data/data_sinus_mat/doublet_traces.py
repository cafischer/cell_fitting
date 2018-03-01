from __future__ import division
import os
import numpy as np
from cell_characteristics.analyze_APs import get_AP_onset_idxs
import matplotlib.pyplot as pl
from cell_fitting.data.data_sinus_mat import get_sinus_data_from_mat
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
from cell_characteristics import to_idx
pl.style.use('paper')


if __name__ == '__main__':
    save_dir = './sinus_mat_files'
    animal = 'rat'
    save_dir_img = os.path.join('../plots/plot_sine_stimulus', 'doublets', animal)
    cell_ids = [file_name.split('_')[0].replace('-', '_') for file_name in os.listdir(save_dir)]
    cell_ids = filter(lambda id: check_rat_or_gerbil(id) == animal, cell_ids)
    repetition = 0
    ISI = 10  # ms
    AP_threshold = 0

    count_freq5_all = {}
    count_freq5_doublets = {}

    for cell_id in cell_ids:
        vs, ts, i_injs, t_i_injs, amp1s, amp2s, freq1s, freq2s = get_sinus_data_from_mat(os.path.join(save_dir,
                                                                                              cell_id.replace('_', '-') + '_Sinus_variables.mat'))
        for i_trace in range(len(vs)):
            v, t, i_inj, t_i_inj = vs[0, i_trace][:, repetition], ts[0, i_trace][:, repetition], i_injs[0, i_trace][:, repetition], t_i_injs[0, i_trace][:, repetition]
            amp1, amp2, freq1, freq2 = amp1s[i_trace], amp2s[i_trace], freq1s[i_trace], freq2s[i_trace]
            save_dir_cell = os.path.join('../plots/plot_sine_stimulus', 'doublets', animal, cell_id,
                                        str(freq1) + '_' + str(freq2) + '_' + str(amp1) + '_' + str(amp2))
            sine1_dur = (1.0 / freq1 * 1000) / 2
            dt = t[1] - t[0]
            onset_dur = offset_dur = 500

            if freq2 == 5:
                count_freq5_all[cell_id] = True

            # check for doublets
            AP_onset_idxs = get_AP_onset_idxs(v, AP_threshold)
            ISIs = np.diff(t[AP_onset_idxs])
            doublet_ISI_lidxs = ISIs < ISI
            doublet_ISI_idxs = AP_onset_idxs[:-1][doublet_ISI_lidxs]

            if np.sum(doublet_ISI_lidxs) > 0:

                if freq2 == 5:
                    count_freq5_doublets[cell_id] = True

                # save and plots
                if not os.path.exists(save_dir_cell):
                    os.makedirs(save_dir_cell)

                # plot v
                pl.figure()
                pl.title('freq1: %.2f, freq2: %.2f, amp1: %.2f, amp2: %.1f' % (freq1, freq2, amp1, amp2))
                pl.plot(t, v, c='k', linewidth=1)
                pl.plot(t[doublet_ISI_idxs], v[doublet_ISI_idxs], 'or')
                pl.xlabel('Time (ms)')
                pl.ylabel('Membrane Potential (mV)')
                pl.tight_layout()
                pl.savefig(os.path.join(save_dir_cell, 'v.png'))
                #pl.show()

                # periods
                period = to_idx(1./freq2*1000, dt)
                start_period = 0
                period_half = to_idx(period, 2)
                period_fourth = to_idx(period, 4)
                onset_idx = to_idx(onset_dur, dt)
                offset_idx = to_idx(onset_dur, dt)
                period_starts = range(len(t))[onset_idx - period_fourth:-offset_idx:period]
                period_ends = range(len(t))[onset_idx + period_half + period_fourth:-offset_idx:period]
                period_starts = period_starts[:len(period_ends)]

                # plot periods one under another
                colors = pl.cm.get_cmap('Greys')(np.linspace(0.2, 1.0, len(period_starts)))
                pl.figure()
                for i, (s, e) in enumerate(zip(period_starts, period_ends)):
                    pl.plot(t[:e - s], v[s:e] + i * -10.0, c=colors[i], label=i, linewidth=1)
                pl.yticks([])
                pl.xlabel('Time (ms)')
                pl.ylabel('Membrane Potential (mV)')
                #pl.xlim(0, 200)
                #pl.ylim(-325, -45)
                pl.legend(fontsize=6, title='Period')
                pl.tight_layout()
                pl.savefig(os.path.join(save_dir_cell, 'periods.png'))
                #pl.show()

                # TODO: mark short ISIs

    print float(len(count_freq5_doublets.keys())) / len(count_freq5_all.keys()) * 100.
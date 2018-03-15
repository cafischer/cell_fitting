import os
import matplotlib.pyplot as pl
import numpy as np
from cell_fitting.data.data_sinus_mat import find_sine_trace_of_cell
from cell_characteristics import to_idx
pl.style.use('paper')


if __name__ == '__main__':
    save_dir = './sinus_mat_files'
    save_dir_img = '../plots/sine_stimulus/doublets/traces/'
    circle = True
    if circle:
        cell_ids = ['2015-08-10f', '2015-08-20d', '2015-08-21b', '2015-08-21f', '2015-08-25b', '2015-08-25d',
                    '2015-08-25e', '2015-08-26b', '2015-08-26e', '2015-08-27d']
    else:
        cell_ids = ['2015-08-05a', '2015-08-05b', '2015-08-10g', '2015-08-11b', '2015-08-11e', '2015-08-20b',
                    '2015-08-20c', '2015-08-20e', '2015-08-20j', '2015-08-27b']

    repetition = 0
    offset = 500
    freq1 = 0.1  # 0.25; 0.5; 0.1
    freq2 = 20  # 5; 20

    save_dir_img = os.path.join(save_dir_img, 'freq1_%.2f_freq2_%.2f' % (freq1, freq2))
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    v_all = []
    t_all = []
    cell_ids_all = []
    amp1s_all = []
    amp2s_all = []
    for cell_id in cell_ids:
        print cell_id
        v_mat, t_mat, amp1s, amp2s, _, _ = find_sine_trace_of_cell(cell_id, None, None, freq1, freq2, repetition, save_dir)
        v_all.extend(v_mat)
        t_all.extend(t_mat)
        amp1s_all.extend(amp1s)
        amp2s_all.extend(amp2s)
        cell_ids_all.extend([cell_id] * len(v_mat))

    v_all = np.vstack(v_all)
    t_all = np.vstack(t_all)
    dt = t_all[0, 1] - t_all[0, 0]
    v_all = v_all[:, to_idx(offset, dt):-to_idx(offset, dt)]
    t_all = t_all[:, to_idx(offset, dt):-to_idx(offset, dt)]

    fig, ax = pl.subplots(len(v_all), 1, sharex='all', sharey='all', figsize=(21, 6 * len(v_all)))  # 29.7
    for i, (v, t) in enumerate(zip(v_all, t_all)):
        ax[i].set_title(cell_ids_all[i])
        ax[i].plot(t, v, 'k', label='$amp_{slow}$: %.2f \n$amp_{fast}$: %.2f ' % (amp1s_all[i], amp2s_all[i]))
        ax[i].set_ylim(-100, 40)
        ax[i].legend(fontsize=10)
        #ax[i].set_ylabel('Membrane Potential (mV)')
        #ax[i].set_xlabel('Time (ms)')
    pl.tight_layout()
    if circle:
        pl.savefig(os.path.join(save_dir_img, 'circle_cells.pdf'))
    else:
        pl.savefig(os.path.join(save_dir_img, 'triangle_cells.pdf'))
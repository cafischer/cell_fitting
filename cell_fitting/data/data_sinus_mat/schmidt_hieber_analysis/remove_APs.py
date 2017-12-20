from __future__ import division
import numpy as np
import os
import json
from grid_cell_stimuli.remove_APs import remove_APs, plot_v_APs_removed
from cell_fitting.data.data_sinus_mat import find_sine_trace


if __name__ == '__main__':
    folder = 'APs_removed'
    save_dir = os.path.join('./results', 'sinus_stimulus')

    # parameters
    AP_threshold = 0
    t_before = 3
    t_after = 6
    params = {'AP_threshold': AP_threshold, 't_before': t_before, 't_after': t_after}

    # load
    amp1 = 0.4
    amp2 = 0.2
    freq1 = 0.1
    freq2 = 5
    sine_folder = str(amp1)+'_'+str(amp2)+'_'+str(freq1)+'_'+str(freq2)
    onset_dur = offset_dur = 500
    v_mat, t_mat, cell_ids, amp1s, amp2s, freq1s, freq2s = find_sine_trace(amp1, amp2, freq1, freq2,
                                                                           save_dir='../sinus_mat_files')

    for v, t, cell_id, amp1, amp2 in zip(v_mat, t_mat, cell_ids, amp1s, amp2s):
        dt = t[1] - t[0]

        # remove APs
        v_APs_removed = remove_APs(v, t, AP_threshold, t_before, t_after)

        # save and plot
        save_dir_cell = os.path.join(save_dir, sine_folder, cell_id, folder)
        if not os.path.exists(save_dir_cell):
            os.makedirs(save_dir_cell)

        np.save(os.path.join(save_dir_cell, 'v.npy'), v_APs_removed)
        np.save(os.path.join(save_dir_cell, 't.npy'), t)

        with open(os.path.join(save_dir_cell, 'params'), 'w') as f:
            json.dump(params, f)

        plot_v_APs_removed(v_APs_removed, v, t, save_dir_cell, show=False)

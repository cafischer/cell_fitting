from __future__ import division
import numpy as np
import os
import json
from grid_cell_stimuli.downsample import antialias_and_downsample, plot_v_downsampled, plot_filter
from cell_fitting.data.data_sinus_mat import find_sine_trace


if __name__ == '__main__':

    amp1 = 0.4
    amp2 = 0.2
    freq1 = 0.1
    freq2 = 5
    sine_folder = str(amp1)+'_'+str(amp2)+'_'+str(freq1)+'_'+str(freq2)
    v_mat, t_mat, cell_ids, amp1s, amp2s, freq1s, freq2s = find_sine_trace(amp1, amp2, freq1, freq2,
                                                                           save_dir='../sinus_mat_files')

    for cell_id in cell_ids:
        save_dir = os.path.join('./results', 'sinus_stimulus', sine_folder, cell_id, 'downsampled')
        save_dir_data = os.path.join('./results', 'sinus_stimulus', sine_folder, cell_id, 'APs_removed')

        # parameters
        cutoff_freq = 2000  # Hz
        dt_new_max = 1. / cutoff_freq * 1000  # ms
        transition_width = 5.0  # Hz
        ripple_attenuation = 60.0  # db
        params = {'dt_new_max': dt_new_max, 'cutoff_freq': cutoff_freq, 'transition_width': transition_width,
                  'ripple_attenuation': ripple_attenuation}

        # load
        v = np.load(os.path.join(save_dir_data, 'v.npy'))
        t = np.load(os.path.join(save_dir_data, 't.npy'))
        dt = t[1] - t[0]

        # downsample
        v_downsampled, t_downsampled, filter = antialias_and_downsample(v, dt, ripple_attenuation, transition_width,
                                                                        cutoff_freq, dt_new_max)

        # save and plot
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.save(os.path.join(save_dir, 'v.npy'), v_downsampled)
        np.save(os.path.join(save_dir, 't.npy'), t_downsampled)

        with open(os.path.join(save_dir, 'params'), 'w') as f:
            json.dump(params, f)

        plot_v_downsampled(v, t, v_downsampled, t_downsampled, save_dir, show=False)
        # plot_filter(filter, dt, save_dir)

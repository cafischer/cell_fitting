from __future__ import division
import numpy as np
import os
import json
from grid_cell_stimuli.ramp_and_theta import get_ramp_and_theta, plot_filter, plot_spectrum, plot_v_ramp_theta
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
        save_dir = os.path.join('./results', 'sinus_stimulus', sine_folder, cell_id, 'ramp_and_theta')
        save_dir_data = os.path.join('./results', 'sinus_stimulus', sine_folder, cell_id, 'downsampled')

        # parameters
        cutoff_ramp = 3  # Hz
        cutoff_theta_low = 5  # Hz
        cutoff_theta_high = 11  # Hz
        transition_width = 1  # Hz
        ripple_attenuation = 60.0  # db
        params = {'cutoff_ramp': cutoff_ramp, 'cutoff_theta_low': cutoff_theta_low,
                  'cut_off_theta_high': cutoff_theta_high, 'transition_width': transition_width,
                  'ripple_attenuation': ripple_attenuation}

        # load
        v = np.load(os.path.join(save_dir_data, 'v.npy'))
        t = np.load(os.path.join(save_dir_data, 't.npy'))
        dt = t[1] - t[0]
        dt_sec = dt / 1000

        # get ramp and theta
        ramp, theta, t_ramp_theta, filter_ramp, filter_theta = get_ramp_and_theta(v, dt, ripple_attenuation,
                                                                                  transition_width, cutoff_ramp,
                                                                                  cutoff_theta_low,
                                                                                  cutoff_theta_high)

        # save and plot
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print 'Ramp_{max-min}: %.2f' % (np.max(ramp) - np.min(ramp))
        print 'Theta_{max-min}: %.2f' % (np.max(theta) - np.min(theta))

        np.save(os.path.join(save_dir, 'ramp.npy'), ramp)
        np.save(os.path.join(save_dir, 'theta.npy'), theta)
        np.save(os.path.join(save_dir, 't.npy'), t_ramp_theta)

        with open(os.path.join(save_dir, 'params'), 'w') as f:
            json.dump(params, f)

        plot_filter(filter_ramp, filter_theta, dt, save_dir)
        plot_spectrum(v, ramp, theta, dt, save_dir)
        plot_v_ramp_theta(v, t, ramp, theta, t_ramp_theta, save_dir, show=False)
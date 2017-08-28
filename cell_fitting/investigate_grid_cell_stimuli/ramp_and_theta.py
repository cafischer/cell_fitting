from __future__ import division
import numpy as np
import os
import json
from grid_cell_stimuli.ramp_and_theta import get_ramp_and_theta, plot_filter, plot_spectrum, plot_v_ramp_theta


if __name__ == '__main__':
    folder = 'test0'
    save_dir = './results/' + folder + '/ramp_and_theta'
    save_dir_data = './results/' + folder + '/downsampled'
    # save_dir_data = './results/test0/APs_removed'

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

    np.save(os.path.join(save_dir, 'ramp.npy'), ramp)
    np.save(os.path.join(save_dir, 'theta.npy'), theta)
    np.save(os.path.join(save_dir, 't.npy'), t_ramp_theta)

    with open(os.path.join(save_dir, 'params'), 'w') as f:
        json.dump(params, f)

    plot_filter(filter_ramp, filter_theta, dt, save_dir)
    plot_spectrum(v, ramp, theta, dt, save_dir)
    plot_v_ramp_theta(v, t, ramp, theta, t_ramp_theta, save_dir)
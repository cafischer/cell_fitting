from __future__ import division
import numpy as np
import os
import json
from grid_cell_stimuli.downsample import antialias_and_downsample, plot_v_downsampled, plot_filter


if __name__ == '__main__':

    folder = 'test0'
    save_dir = './results/'+folder+'/downsampled'
    save_dir_data = './results/'+folder+'/APs_removed'

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

    plot_v_downsampled(v, t, v_downsampled, t_downsampled, save_dir)
    plot_filter(filter, dt, save_dir)

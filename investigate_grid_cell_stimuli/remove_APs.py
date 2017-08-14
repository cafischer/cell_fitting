from __future__ import division
import numpy as np
import os
import json
from grid_cell_stimuli.remove_APs import remove_APs, plot_v_APs_removed


if __name__ == '__main__':
    folder = 'test0'
    save_dir = './results/'+folder+'/APs_removed'
    save_dir_data = './results/'+folder+'/data'

    # parameters
    AP_threshold = -20
    t_before = 3
    t_after = 6
    params = {'AP_threshold': AP_threshold, 't_before': t_before, 't_after': t_after}

    # load
    v = np.load(os.path.join(save_dir_data, 'v.npy'))
    t = np.load(os.path.join(save_dir_data, 't.npy'))
    dt = t[1] - t[0]

    # remove APs
    v_APs_removed = remove_APs(v, t, AP_threshold, t_before, t_after)

    # save and plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'v.npy'), v_APs_removed)

    with open(os.path.join(save_dir, 'params'), 'w') as f:
        json.dump(params, f)

    plot_v_APs_removed(v_APs_removed, v, t, save_dir)

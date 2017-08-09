from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import copy
import os
import json
from cell_characteristics.analyze_APs import get_AP_onsets


if __name__ == '__main__':
    save_dir = './results/test0/APs_removed'
    save_dir_data = './results/test0/data'

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
    AP_onsets = get_AP_onsets(v, threshold=AP_threshold)
    dur_AP = t_after + t_before
    idx_before = int(round(t_before / dt))
    idx_after = int(round(t_after / dt))
    v_APs_removed = copy.copy(v)
    for onset in AP_onsets:
        slope = (v[onset+idx_after] - v[onset-idx_before]) / dur_AP
        v_APs_removed[onset-idx_before:onset+idx_after+1] = slope \
                                                            * np.arange(0, dur_AP + dt, dt) + v[onset - idx_before]

    # save and plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'v.npy'), v_APs_removed)
    np.save(os.path.join(save_dir, 't.npy'), t)

    with open(os.path.join(save_dir, 'params'), 'w') as f:
        json.dump(params, f)

    pl.figure()
    pl.plot(t, v, 'k', label='$V$')
    pl.plot(t, v_APs_removed, 'b', label='$V_{APs\ removed}$')
    pl.ylabel('Membrane potential (mV)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'v_APs_removed.svg'))
    pl.show()

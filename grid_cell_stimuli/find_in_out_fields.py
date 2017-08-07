from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
import pandas as pd
import json
import random
from cell_characteristics.analyze_APs import get_AP_onsets, get_AP_max_idx


def get_firing_rate_per_bin(APs, t, position, bins):
    run_start_idx = np.where(np.diff(position) < 0)[0]
    APs_runs = np.split(APs, run_start_idx)
    pos_runs = np.split(position, run_start_idx)
    t_runs = np.split(t, run_start_idx)
    firing_rate_per_run = np.zeros((len(run_start_idx) + 1, len(bins)))
    max_diff = lambda x: np.max(x) - np.min(x)
    for i_run, (APs_run, pos_run, t_run) in enumerate(zip(APs_runs, pos_runs, t_runs)):
        pos_binned = np.digitize(pos_run, bins)
        AP_count_per_bin = pd.Series(APs_run).groupby(pos_binned).sum()
        seconds_per_bin = pd.Series(t_run).groupby(pos_binned).apply(max_diff) / 1000
        firing_rate_per_run[i_run, np.unique(pos_binned)] = AP_count_per_bin / seconds_per_bin

        # for testing: print AP_max_idx[0] in np.where(pos_binned == AP_count_per_bin.index[AP_count_per_bin > 0][0])[0]
    # pl.figure()
    # for i_run in range(len(t_runs)):
    #     pl.plot(firing_rate_per_run[i_run, :])
    # pl.show()

    firing_rate = np.mean(firing_rate_per_run, 0)
    return firing_rate, firing_rate_per_run


def get_v_per_run(v, t, position):
    run_start_idx = np.where(np.diff(position) < 0)[0]
    v_runs = np.split(v, run_start_idx)
    t_runs = np.split(t, run_start_idx)
    return v_runs, t_runs


def shuffle_APs(APs, n_shuffles, seed):
    n_samples = len(v)
    APs_shuffles = np.zeros((n_shuffles, n_samples))
    random_generator = random.Random()
    random_generator.seed(seed)
    for i in range(n_shuffles):
        idx = random_generator.randint(int(np.ceil(0.05 * n_samples)), int(np.floor(0.95 * n_samples)))
        APs_shuffles[i, :] = np.concatenate((APs[idx:], APs[:idx]))
    return APs_shuffles


def get_start_end_ones(x):
    start = np.where(np.diff(x) == 1)[0] + 1
    end = np.where(np.diff(x) == -1)[0]
    if x[0] == 1:
        start = np.concatenate((np.array([0]), start))
    if x[-1] == 1:
        end = np.concatenate((end, np.array([len(x) - 1])))
    return start, end


if __name__ == '__main__':
    save_dir = './results/test0/in_out_fields'
    save_dir_data = './results/test0/data'

    # parameters
    seed = 1
    n_shuffles = 100
    bin_size = 5  # cm
    params = {'seed': seed, 'n_shuffles': n_shuffles, 'bin_size': bin_size}

    # load
    v = np.load(os.path.join(save_dir_data, 'v.npy'))
    t = np.load(os.path.join(save_dir_data, 't.npy'))
    position = np.load(os.path.join(save_dir_data, 'position.npy'))
    dt = t[1] - t[0]

    # compute spike train
    AP_onsets = get_AP_onsets(v, threshold=-20)
    AP_onsets = np.concatenate((AP_onsets, np.array([len(v)])))
    AP_max_idx = [get_AP_max_idx(v, AP_onsets[i], AP_onsets[i + 1], interval=int(round(2/dt))) for i in
                  range(len(AP_onsets) - 1)]
    APs = np.zeros(len(v))
    APs[AP_max_idx] = 1

    # shuffle
    APs_shuffles = shuffle_APs(APs, n_shuffles, seed)

    # bin according to position (bin=5cm) and compute firing rate
    bins = np.arange(bin_size, np.max(position) + bin_size, bin_size)

    firing_rate_real, firing_rate_per_run = get_firing_rate_per_bin(APs, t, position, bins)
    firing_rate_shuffled = np.zeros((n_shuffles, len(bins)))
    for i, APs_shuffled in enumerate(APs_shuffles):
        firing_rate_shuffled[i, :], _ = get_firing_rate_per_bin(APs_shuffled, t, position, bins)

    # compute P-value: percent of shuffled firing rates that were higher than the cells real firing rate
    p_value = np.array([np.sum(firing_rate_shuffled[:, i] > firing_rate_real[i]) / n_shuffles
                        for i in range(len(bins))])

    # get in-field and out-fields
    out_field = np.zeros(len(bins))
    out_field[1 - p_value <= 0.05] = 1  # 1 - P value <= 0.05
    idx1 = np.where(out_field)[0]
    groups = np.split(idx1, np.where(np.diff(idx1) > 1)[0]+1)
    for g in groups:
        if len(g) <= 2:  # more than 2 adjacent bins
            out_field[g] = 0

    in_field = np.zeros(len(bins))
    in_field[1 - p_value >= 0.85] = 1  # 1 - P value >= 0.85
    idx1 = np.where(in_field)[0]
    groups = np.split(idx1, np.where(np.diff(idx1) > 1)[0]+1)
    for g in groups:
        if len(g) <= 3:  # more than 3 adjacent bins
            in_field[g] = 0
        else:  # extend by 1 bin left and right if: 1 - P value >= 0.70
            n_runs = np.shape(firing_rate_per_run)[0]
            spiked_per_run = [np.sum(firing_rate_per_run[i, g]) > 0 for i in range(n_runs)]
            if np.sum(spiked_per_run) / n_runs < 0.2:  # spikes on at least 20 % of all runs
                in_field[g] = 0
                continue
            if g[0] - 1 > 0:
                if 1 - p_value[g[0]-1] >= 0.7:
                    in_field[g[0]-1] = 1
            if g[-1] + 1 < len(bins):
                if 1 - p_value[g[-1]+1] >= 0.7:
                    in_field[g[-1]+1] = 1

    # save and plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'in_field.npy'), in_field)
    np.save(os.path.join(save_dir, 'out_field.npy'), out_field)
    np.save(os.path.join(save_dir, 'firing_rate_real.npy'), firing_rate_real)
    np.save(os.path.join(save_dir, 'firing_rate_shuffled.npy'), firing_rate_shuffled)

    with open(os.path.join(save_dir, 'params.json'), 'w') as f:
        json.dump(params, f)

    pl.figure()
    pl.plot(firing_rate_real, 'k', label='Real')
    pl.plot(np.mean(firing_rate_shuffled, 0), 'r', label='Shuffled mean')
    pl.xticks(np.arange(0, len(bins)+1, len(bins)/4), np.arange(0, bins[-1]+bins[-1]/4, bins[-1]/4))
    pl.xlabel('Position (cm)', fontsize=16)
    pl.ylabel('Firing rate (spikes/sec)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'firing_rate_binned.png'))
    pl.show()

    start_out, end_out = get_start_end_ones(out_field)
    start_in, end_in = get_start_end_ones(in_field)

    pl.figure()
    pl.plot(1 - p_value, 'g', label='1 - P value')
    for i, (s, e) in enumerate(zip(start_out, end_out)):
        pl.hlines(-0.01, s, e, 'b', label='Out field' if i==0 else None)
    for i, (s, e) in enumerate(zip(start_in, end_in)):
        pl.hlines(-0.01, s, e, 'r', label='In field' if i==0 else None)
    pl.xticks(np.arange(0, len(bins)+1, len(bins)/4), np.arange(0, bins[-1]+bins[-1]/4, bins[-1]/4))
    pl.xlabel('Position (cm)', fontsize=16)
    pl.ylabel('Firing rate (spikes/sec)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'p_value.png'))
    pl.show()

    pl.figure()
    pl.plot(firing_rate_real, 'k', label='Firing rate')
    for i, (s, e) in enumerate(zip(start_out, end_out)):
        pl.hlines(-0.01, s, e, 'b', label='Out field' if i==0 else None)
    for i, (s, e) in enumerate(zip(start_in, end_in)):
        pl.hlines(-0.01, s, e, 'r', label='In field' if i==0 else None)
    pl.xticks(np.arange(0, len(bins)+1, len(bins)/4), np.arange(0, bins[-1]+bins[-1]/4, bins[-1]/4))
    pl.xlabel('Position (cm)', fontsize=16)
    pl.ylabel('Firing rate (spikes/sec)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'firing_rate_and_fields.png'))
    pl.show()

    v_per_run, t_per_run = get_v_per_run(v, t, position)
    i_run = 0
    start_out = start_out / len(bins) * t_per_run[i_run][-1]
    end_out = end_out / len(bins) * t_per_run[i_run][-1]
    start_in = start_in / len(bins) * t_per_run[i_run][-1]
    end_in = end_in / len(bins) * t_per_run[i_run][-1]

    pl.figure()
    pl.plot(t_per_run[i_run], v_per_run[i_run], 'k', label='Membrane potential', linewidth=2)
    for i, (s, e) in enumerate(zip(start_out, end_out)):
        pl.hlines(np.min(v_per_run[i_run])-1, s, e, 'b', label='Out field' if i==0 else None)
    for i, (s, e) in enumerate(zip(start_in, end_in)):
        pl.hlines(np.min(v_per_run[i_run])-1, s, e, 'r', label='In field' if i==0 else None)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.ylabel('Firing rate (spikes/sec)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'v_and_fields.png'))
    pl.show()
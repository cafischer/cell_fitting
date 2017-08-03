import numpy as np
import matplotlib.pyplot as pl
import os
import pandas as pd
import json
from cell_characteristics.analyze_APs import get_AP_onsets, get_AP_max_idx


def get_firing_rate_per_bin(APs, t, position, bins):
    run_start_idx = np.where(np.diff(position) < 0)[0]
    APs_runs = np.split(APs, run_start_idx)
    pos_runs = np.split(position, run_start_idx)
    t_runs = np.split(t, run_start_idx)
    firing_rate_per_bin_per_run = np.zeros((len(run_start_idx) + 1, len(bins)))
    max_diff = lambda x: np.max(x) - np.min(x)
    for i_run, (APs_run, pos_run, t_run) in enumerate(zip(APs_runs, pos_runs, t_runs)):
        pos_binned = np.digitize(pos_run, bins)
        AP_count_per_bin = pd.Series(APs_run).groupby(pos_binned).sum()
        seconds_per_bin = pd.Series(t_run).groupby(pos_binned).apply(max_diff)
        firing_rate_per_bin_per_run[i_run, np.unique(pos_binned)] = AP_count_per_bin / seconds_per_bin

        # for testing: print AP_max_idx[0] in np.where(pos_binned == AP_count_per_bin.index[AP_count_per_bin > 0][0])[0]
    firing_rate_per_bin = np.mean(firing_rate_per_bin_per_run, 0)
    return firing_rate_per_bin, firing_rate_per_bin_per_run


def get_v_per_run(v, t, position):
    run_start_idx = np.where(np.diff(position) < 0)[0]
    v_runs = np.split(v, run_start_idx)
    t_runs = np.split(t, run_start_idx)
    return v_runs, t_runs


if __name__ == '__main__':
    save_dir = './results/test0/in_out_fields'
    save_dir_data = './results/test0/data'

    # parameters
    n_shuffles = 100
    bin_size = 5  # cm
    # TODO: save parameters

    # load
    v = np.load(os.path.join(save_dir_data, 'v.npy'))
    t = np.load(os.path.join(save_dir_data, 't.npy'))
    dt = t[1] - t[0]
    dur_run = json.load(os.path.join(save_dir_data, 'params'))['t_run']

    # compute spike train
    AP_onsets = get_AP_onsets(v, threshold=-20)
    AP_onsets = np.concatenate((AP_onsets, np.array([len(v)])))
    AP_max_idx = [get_AP_max_idx(v, AP_onsets[i], AP_onsets[i + 1], interval=2 / dt) for i in
                  range(len(AP_onsets) - 1)]
    APs = np.zeros(len(v))
    APs[AP_max_idx] = 1

    # shuffle
    n_samples = len(v)
    APs_shuffles = np.zeros((n_shuffles, n_samples))
    for i in range(n_shuffles):
        idx = np.random.randint(int(np.ceil(0.05*n_samples)), int(np.floor(0.95 * n_samples)))
        APs_shuffles[i, :] = np.concatenate((APs[idx:], APs[:idx]))

    # bin according to position (bin=5cm) and compute firing rate
    v_animal = 40  # cm/sec
    position = (t % dur_run) * v_animal
    bins = np.arange(bin_size, np.max(position) + bin_size, bin_size)

    firing_rate_per_bin_real, firing_rate_per_run = get_firing_rate_per_bin(APs, t, position, bins)
    firing_rate_per_bin_shuffled = np.zeros((n_shuffles, len(bins)))
    for i, APs_shuffled in enumerate(APs_shuffles):
        firing_rate_per_bin_shuffled[i, :], _ = get_firing_rate_per_bin(APs_shuffled, t, position, bins)

    # compute P-value: percent of shuffled firing rates that were higher than the cells real firing rate
    p_value_per_bin = np.array([np.sum(firing_rate_per_bin_shuffled[:, i] > firing_rate_per_bin_real[i])
                                for i in range(len(bins))])

    # get in-field and out-fields
    out_field = np.zeros(len(bins))
    out_field[1 - p_value_per_bin <= 0.05] = 1  # 1 - P value <= 0.05

    in_field = np.zeros(len(bins))
    in_field[1 - p_value_per_bin >= 0.85] = 1  # 1 - P value >= 0.85
    idx1 = np.where(in_field)[0]
    groups = np.split(idx1, np.where(np.diff(idx1) > 1)[0]+1)
    for g in groups:
        if len(g) < 4:  # > 3 bins
            in_field[g] = 0
        else:  # extend by 1 bin left and right if: 1 - P value >= 0.70
            n_runs = np.shape(firing_rate_per_run)[0]
            spiked_per_run = [np.sum(firing_rate_per_run[i, g]) > 0 for i in range(n_runs)]
            if np.sum(spiked_per_run) / n_runs < 0.2:  # spikes on at least 20 % of all runs
                in_field[g] = 0
                continue
            if g[0] - 1 > 0:
                if 1 - p_value_per_bin[g[0]-1] >= 0.7:
                    in_field[g[0]-1] = 1
            if g[-1] + 1 < len(bins):
                if 1 - p_value_per_bin[g[-1]+1] >= 0.7:
                    in_field[g[-1]+1] = 1

    # save and plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'in_field.npy'), in_field)
    np.save(os.path.join(save_dir, 'out_field.npy'), out_field)

    pl.figure()
    pl.plot(firing_rate_per_bin_real, 'k', label='Real')
    pl.plot(np.mean(firing_rate_per_bin_shuffled, 0), 'r', label='Shuffled mean')
    pl.xlabel('Bin', fontsize=16)
    pl.ylabel('Firing rate (spikes/sec)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'firing_rate_binned.png'))
    pl.show()

    pl.figure()
    pl.plot(firing_rate_per_bin_real, 'k', label='Firing rate')
    start_out = np.where(out_field == 1)[0]
    end_out = np.where(out_field == -1)[0]
    for s, e in zip(start_out, end_out):
        pl.hlines(np.min(v)-1, s, e, 'b')
    start_in = np.where(in_field == 1)[0]
    end_in = np.where(in_field == -1)[0]
    for i, (s, e) in enumerate(zip(start_out, end_out)):
        pl.hlines(np.min(v)-1, s, e, 'b', label='Out field' if i==0 else None)
    for s, e in zip(start_in, end_in):
        pl.hlines(np.min(v)-1, s, e, 'r', label='In field' if i==0 else None)
    pl.xlabel('Bin', fontsize=16)
    pl.ylabel('Firing rate (spikes/sec)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'firing_rate_and_fields.png'))
    pl.show()

    v_per_run, t_per_run = get_v_per_run(v, t, position)
    pl.figure()
    pl.plot(t_per_run[0], v_per_run[0], 'k', label='Membrane potential', linewidth=2)
    start_out = np.where(out_field == 1)[0] / len(bins) * t_per_run[0][-1]
    end_out = np.where(out_field == -1)[0] / len(bins) * t_per_run[0][-1]
    for s, e in zip(start_out, end_out):
        pl.hlines(np.min(v)-1, s, e, 'b')
    start_in = np.where(in_field == 1)[0] / len(bins) * t_per_run[0][-1]
    end_in = np.where(in_field == -1)[0] / len(bins) * t_per_run[0][-1]
    for i, (s, e) in enumerate(zip(start_out, end_out)):
        pl.hlines(np.min(v)-1, s, e, 'b', label='Out field' if i==0 else None)
    for s, e in zip(start_in, end_in):
        pl.hlines(np.min(v)-1, s, e, 'r', label='In field' if i==0 else None)
    pl.xlabel('Bin', fontsize=16)
    pl.ylabel('Firing rate (spikes/sec)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'v_and_fields.png'))
    pl.show()
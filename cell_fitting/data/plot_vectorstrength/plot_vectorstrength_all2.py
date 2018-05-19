from __future__ import division
import numpy as np
from scipy.io import loadmat
from scipy.signal import vectorstrength
import matplotlib.pyplot as pl
import os
import copy
import json


if __name__ == '__main__':
    save_dir = './vectorstrength/all_ISIs'
    file_names = np.array(os.listdir(save_dir))

    durs_sine_slow = ['1s', '2s', '5s', '10s']
    relative_tos = ['fast']  # 'slow'
    freqs_fast = ['RF', '5Hz', '20Hz']

    # just initialization of dicts
    vectorstrength_dict = {}
    n_cells_dict = {}
    phases_dict = {}
    phases_dict_cells = {}
    for relative_to in relative_tos:
        vectorstrength_dict[relative_to] = {}
        n_cells_dict[relative_to] = {}
        phases_dict[relative_to] = {}
        phases_dict_cells[relative_to] = {}
    for relative_to in relative_tos:
        for dur_sine_slow in durs_sine_slow:
            for freq_fast in freqs_fast:
                vectorstrength_dict[relative_to][dur_sine_slow+'_'+freq_fast] = []
                n_cells_dict[relative_to][dur_sine_slow+'_'+freq_fast] = 0
                phases_dict[relative_to][dur_sine_slow + '_' + freq_fast] = []
                phases_dict_cells[relative_to][dur_sine_slow + '_' + freq_fast] = {}

    for file_name in file_names:
        cell_id = file_name.split('_')[0].replace('-', '_')
        mat = loadmat(os.path.join(save_dir, file_name))['short_ISI']

        for relative_to in relative_tos:
            for freq_fast in freqs_fast:
                for dur_sine_slow in durs_sine_slow:

                    stim_lens = mat[:, 6]
                    freqs = mat[:, 7]

                    if freq_fast == 'RF':
                        freqs_fast_without_RF = copy.copy(freqs_fast)
                        freqs_fast_without_RF.remove('RF')
                        mat_selected = mat[np.logical_and(stim_lens == float(dur_sine_slow[:-1]),
                                                                         ~np.any(np.vstack(
                                                                         [freqs == float(f[:-2]) for f
                                                                          in freqs_fast_without_RF]), axis=0))]
                    else:
                        mat_selected = mat[np.logical_and(stim_lens == float(dur_sine_slow[:-1]),
                                                                     freqs == float(freq_fast[:-2]))]

                    phases = mat_selected[:, 12]
                    if len(phases) > 1:
                        vs = vectorstrength(phases, 360)[0]  # 0 strength, 1 phase
                        vectorstrength_dict[relative_to][dur_sine_slow+'_'+freq_fast].append(vs)
                        n_cells_dict[relative_to][dur_sine_slow+'_'+freq_fast] += 1
                        phases_dict[relative_to][dur_sine_slow + '_' + freq_fast].extend(np.array(phases, dtype=float))
                        phases_dict_cells[relative_to][dur_sine_slow + '_' + freq_fast][cell_id] = np.array(phases, dtype=float).tolist()

    # save dicts
    with open('./vectorstrength_dict_all.json', 'w') as f:
        json.dump(vectorstrength_dict, f)
    with open('./n_cells_dict_all.json', 'w') as f:
        json.dump(n_cells_dict, f)
    with open('./phases_dict_all.json', 'w') as f:
        json.dump(phases_dict, f)
    with open('./phases_dict_cells_all.json', 'w') as f:
        json.dump(phases_dict_cells, f)

    # plot
    bins = np.arange(0, 1+0.025, 0.025)
    for relative_to in relative_tos:

        fig, axes = pl.subplots(len(freqs_fast), len(durs_sine_slow), sharex='all', sharey='all', figsize=(12, 7))
        for i, freq_fast in enumerate(freqs_fast):
            for j, dur_sine_slow in enumerate(durs_sine_slow):

                if len(vectorstrength_dict[relative_to][dur_sine_slow+'_'+freq_fast]) == 0:
                    axes[len(freqs_fast)-i-1, j].axis('off')

                axes[len(freqs_fast)-i-1, j].hist(vectorstrength_dict[relative_to][dur_sine_slow+'_'+freq_fast],
                                bins=bins,
                                weights=np.ones(len(vectorstrength_dict[relative_to][dur_sine_slow+'_'+freq_fast]))
                                        / n_cells_dict[relative_to][dur_sine_slow+'_'+freq_fast], color='0.5')
                l1 = axes[len(freqs_fast) - i - 1, j].axvline(
                    np.mean(vectorstrength_dict[relative_to][dur_sine_slow+'_'+freq_fast]),
                    0, 1, color='r', label='')
                l2 = axes[len(freqs_fast) - i - 1, j].axvline(
                    vectorstrength(phases_dict[relative_to][dur_sine_slow + '_' + freq_fast], 360)[0],
                    0, 1, color='b', label='')
                axes[len(freqs_fast) - i - 1, j].set_xlim(0, 1)
                axes[len(freqs_fast) - i - 1, j].spines['top'].set_visible(False)
                axes[len(freqs_fast) - i - 1, j].spines['right'].set_visible(False)

                if (len(freqs_fast)-i-1) == 0:
                    axes[len(freqs_fast) - i - 1, j].set_title(dur_sine_slow, fontsize=16)
                    #axes[len(freqs_fast)-i-1, j].set_xlabel(dur_sine_slow, fontsize=16)
                    #axes[len(freqs_fast)-i-1, j].xaxis.set_label_position("top")
                if j == len(freqs_fast):
                    axes[len(freqs_fast)-i-1, j].set_ylabel(freq_fast, fontsize=16)
                    axes[len(freqs_fast)-i-1, j].yaxis.set_label_position("right")
        fig.text(0.01, 0.5, 'Vector strength relative to '+relative_to+' oscillation (all APs)', va='center',
                 rotation='vertical', fontsize=16)
        fig.legend((l1, l2), ('avg.', 'all phases'), loc='upper right')
    pl.tight_layout()
    pl.subplots_adjust(left=0.08)
    pl.savefig(os.path.join('./', 'vector_strength_all.png'))
    pl.show()
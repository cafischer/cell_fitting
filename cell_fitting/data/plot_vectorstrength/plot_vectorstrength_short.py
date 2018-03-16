from __future__ import division
import numpy as np
import json
from scipy.signal import vectorstrength
import matplotlib.pyplot as pl
import os
import pandas as pd


if __name__ == '__main__':
    save_dir = './vectorstrength/Phasen_shortISIs.csv'

    durs_sine_slow = ['1s', '2s', '5s', '10s']
    relative_tos = ['fast']  # 'slow'
    freqs_fast = ['RF', '5Hz', '20Hz']

    # just initialization of dicts
    vectorstrength_dict = {}
    n_cells_dict = {}
    phases_dict = {}
    for relative_to in relative_tos:
        vectorstrength_dict[relative_to] = {}
        n_cells_dict[relative_to] = {}
        phases_dict[relative_to] = {}
    for relative_to in relative_tos:
        for dur_sine_slow in durs_sine_slow:
            for freq_fast in freqs_fast:
                vectorstrength_dict[relative_to][dur_sine_slow+'_'+freq_fast] = []
                n_cells_dict[relative_to][dur_sine_slow+'_'+freq_fast] = 0
                phases_dict[relative_to][dur_sine_slow + '_' + freq_fast] = []

    # read data sheat
    mat = pd.read_csv(save_dir, dtype={'cellname': str, 'ISI': float, 'Stim length': str, 'Freq': str,
                                       'fast Phase': float, 'slow Phase': float})
    cell_ids = np.unique(mat['cellname'].dropna().values)

    for cell_id in cell_ids:
        for relative_to in relative_tos:
            for dur_sine_slow in durs_sine_slow:
                for freq_fast in freqs_fast:

                    mat_cell_id = mat.ix[np.where(mat.cellname == cell_id)[0]]
                    if freq_fast == 'RF':
                        mat_selected = mat_cell_id.ix[np.logical_and(mat_cell_id['Stim length'] == dur_sine_slow[:-1],
                                                                     ~np.any(np.vstack([(mat_cell_id['Freq'] == f[:-2]).values for f in freqs_fast]), axis=0))]
                    else:
                        mat_selected = mat_cell_id.ix[np.logical_and(mat_cell_id['Stim length'] == dur_sine_slow[:-1],
                                                                     mat_cell_id['Freq'] == freq_fast[:-2])]

                    if not mat_selected.empty:
                        phases = mat_selected[relative_to+' Phase'].values

                        vs = vectorstrength(phases, 360)[0]  # 0 strength, 1 phase
                        vectorstrength_dict[relative_to][dur_sine_slow+'_'+freq_fast].append(vs)
                        n_cells_dict[relative_to][dur_sine_slow+'_'+freq_fast] += 1
                        phases_dict[relative_to][dur_sine_slow + '_' + freq_fast].extend(phases)

    # save dicts
    with open('./vectorstrength_dict_short.json', 'w') as f:
        json.dump(vectorstrength_dict, f)
    with open('./n_cells_dict_short.json', 'w') as f:
        json.dump(n_cells_dict, f)
    with open('./phases_dict_short.json', 'w') as f:
        json.dump(phases_dict, f)

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
        fig.text(0.01, 0.5, 'Vector strength relative to '+relative_to+' oscillation (APs with ISI<10ms)', va='center',
                 rotation='vertical', fontsize=16)
        fig.legend((l1, l2), ('avg.', 'all phases'), loc='upper right')
    pl.tight_layout()
    pl.subplots_adjust(left=0.08)
    pl.savefig(os.path.join('./', 'vector_strength_short.png'))
    pl.show()
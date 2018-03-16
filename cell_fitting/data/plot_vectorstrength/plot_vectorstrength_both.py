from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
import json
from scipy.stats import ks_2samp


def get_cumulative_hist(x):
    ISIs_sorted = np.sort(x)
    cum_hist_x, cum_hist_y = np.unique(ISIs_sorted, return_counts=True)
    cum_hist_y = np.cumsum(cum_hist_y) / len(x)
    cum_hist_x = np.insert(cum_hist_x, 0, 0)
    cum_hist_y = np.insert(cum_hist_y, 0, 0)
    return cum_hist_y, cum_hist_x


if __name__ == '__main__':
    save_dir = './vectorstrength/Phase_vs_Frequencies'
    all_file_names = np.array(os.listdir(save_dir))

    durs_sine_slow = ['1s', '2s', '5s', '10s']
    relative_tos = ['fast']  # 'slow'
    freqs_fast = ['RF', '5Hz', '20Hz']

    # load
    with open('./vectorstrength_dict_all.json', 'r') as f:
        vectorstrength_dict_all = json.load(f)
    with open('./n_cells_dict_all.json', 'r') as f:
        n_cells_dict_all = json.load(f)
    with open('./n_cells_dict_all.json', 'r') as f:
        n_cells_dict_all = json.load(f)
    with open('./vectorstrength_dict_short.json', 'r') as f:
        vectorstrength_dict_short = json.load(f)
    with open('./n_cells_dict_short.json', 'r') as f:
        n_cells_dict_short = json.load(f)
    with open('./phases_dict_short.json', 'r') as f:
        phases_dict_short = json.load(f)

    # plot
    max_x = 1.01
    for relative_to in relative_tos:

        fig, axes = pl.subplots(len(freqs_fast), len(durs_sine_slow), sharex='all', sharey='all', figsize=(12, 7))
        for i, freq_fast in enumerate(freqs_fast):
            for j, dur_sine_slow in enumerate(durs_sine_slow):

                if len(vectorstrength_dict_all[relative_to][dur_sine_slow+'_'+freq_fast]) == 0 and \
                        len(vectorstrength_dict_short[relative_to][dur_sine_slow+'_'+freq_fast]) == 0:
                    axes[len(freqs_fast)-i-1, j].axis('off')

                if len(vectorstrength_dict_all[relative_to][dur_sine_slow + '_' + freq_fast]) > 0:
                    cum_hist_y_all, cum_hist_x_all = get_cumulative_hist(vectorstrength_dict_all[relative_to]
                                                                                 [dur_sine_slow+'_'+freq_fast])
                    cum_hist_x_all = np.insert(cum_hist_x_all, len(cum_hist_x_all), max_x)
                    cum_hist_y_all = np.insert(cum_hist_y_all, len(cum_hist_y_all), 1.0)

                    l1, = axes[len(freqs_fast) - i - 1, j].plot(cum_hist_x_all, cum_hist_y_all, drawstyle='steps-post',
                                                                color='0.7', label='all')

                if len(vectorstrength_dict_short[relative_to][dur_sine_slow + '_' + freq_fast]) > 0:
                    cum_hist_y_short, cum_hist_x_short = get_cumulative_hist(vectorstrength_dict_short[relative_to]
                                                                             [dur_sine_slow+'_'+freq_fast])
                    cum_hist_x_short = np.insert(cum_hist_x_short, len(cum_hist_x_short), max_x)
                    cum_hist_y_short = np.insert(cum_hist_y_short, len(cum_hist_y_short), 1.0)

                    l2, = axes[len(freqs_fast) - i - 1, j].plot(cum_hist_x_short, cum_hist_y_short,
                                                                drawstyle='steps-post',
                                                                color='0.3', label='ISI<10ms')


                if len(vectorstrength_dict_all[relative_to][dur_sine_slow+'_'+freq_fast]) > 0 and \
                        len(vectorstrength_dict_short[relative_to][dur_sine_slow+'_'+freq_fast]) > 0:
                    D, p_val = ks_2samp(vectorstrength_dict_all[relative_to][dur_sine_slow+'_'+freq_fast],
                                        vectorstrength_dict_short[relative_to][dur_sine_slow+'_'+freq_fast])
                    axes[len(freqs_fast) - i - 1, j].text(0.3, 0.9, 'KS p-val: %.3f' % p_val, fontsize=10)

                axes[len(freqs_fast) - i - 1, j].set_xlim(0, max_x)
                axes[len(freqs_fast) - i - 1, j].spines['top'].set_visible(False)
                axes[len(freqs_fast) - i - 1, j].spines['right'].set_visible(False)

                if (len(freqs_fast)-i-1) == 0:
                    axes[len(freqs_fast) - i - 1, j].set_title(dur_sine_slow, fontsize=16)
                if j == len(freqs_fast):
                    axes[len(freqs_fast)-i-1, j].set_ylabel(freq_fast, fontsize=16)
                    axes[len(freqs_fast)-i-1, j].yaxis.set_label_position("right")
        fig.text(0.01, 0.5, 'Cumulative vector strength relative to '+relative_to+' oscillation', va='center',
                 rotation='vertical', fontsize=16)
        fig.legend((l1, l2), ('all', 'ISI<10ms'), loc='upper right')
    pl.tight_layout()
    pl.subplots_adjust(left=0.08)
    pl.savefig(os.path.join('./', 'vector_strength_comparison.png'))
    pl.show()
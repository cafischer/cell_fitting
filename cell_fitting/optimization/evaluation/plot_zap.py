from __future__ import division

import matplotlib.pyplot as pl
import numpy as np

pl.style.use('paper')
import os
from cell_fitting.optimization.simulate import iclamp_handling_onset
from nrn_wrapper import Cell
from cell_fitting.read_heka import get_v_and_t_from_heka
from cell_fitting.data import set_v_rest
from cell_characteristics import to_idx
import statsmodels.api as sm
from cell_fitting.optimization.fitfuns import impedance
from cell_fitting.read_heka.i_inj_functions import get_zap



def apply_zap_stimulus(cell, amp=0.1, freq0=0, freq1=20, onset_dur=2000, offset_dur=2000, zap_dur=30000,
                       tstop=34000, dt=0.01):

    i_exp = get_zap(amp, freq0, freq1, onset_dur, offset_dur, zap_dur, tstop, dt)

    # get simulation parameters
    simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': -75, 'tstop': tstop,
                         'dt': dt, 'celsius': 35, 'onset': 200}

    # record v
    v, t, i_exp = iclamp_handling_onset(cell, **simulation_params)

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'zap')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    freqs = lambda x: (freq0-freq1)/(onset_dur-t[-1]-offset_dur) * x \
                      + (freq0-(freq0-freq1)/(onset_dur-t[-1]-offset_dur)*onset_dur)
    heavy_freqs = lambda x: freqs(x) if onset_dur < x < t[-1]-offset_dur else 0
    freqs_out = lambda x: "%.2f" % heavy_freqs(x)


    # exp data
    v_exp, t_exp = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), 'Zap20')
    v_exp = v_exp[0, :]
    t_exp = t_exp[0, :]
    v_exp = set_v_rest(v_exp, v_exp[0], -75)

    # plot
    fig, ax1 = pl.subplots()
    ax2 = ax1.twiny()
    #ax1.plot(t_exp, v_exp, 'k', label='Exp. Data')
    ax1.plot(t, v, 'r', label='Model')
    ax1.set_xlim(0, t[-1])
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xticklabels(map(freqs_out, ax1.get_xticks()))
    ax2.spines['top'].set_visible(True)
    ax2.set_xlabel('Frequency (Hz)')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane potential (mV)')
    #ax1.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'v.png'))
    pl.show()

    # plot
    # import pandas as pd
    # i_exp2 = pd.read_csv('../../data/Protocols/Zap20.csv')
    # fig, ax1 = pl.subplots()
    # ax2 = ax1.twiny()
    # print (t_exp[1]-t_exp[0])
    # pl.plot(np.arange(len(i_exp2))*(t_exp[1]-t_exp[0])*2, i_exp2)
    # #pl.plot(np.arange(len(i_exp2)) * (0.019), i_exp2)
    # pl.plot(t[:-int(round((onset_dur+offset_dur)/dt))], i_exp[int(round(onset_dur/dt)):-int(round(offset_dur/dt))])
    # ax2.set_xticks(ax1.get_xticks())
    # ax2.set_xticklabels(map(freqs_out, ax1.get_xticks()))
    # ax2.set_xlabel('Frequency $(Hz)$', fontsize=16)
    # ax1.set_xlabel('Time $(ms)$', fontsize=16)
    # ax1.set_ylabel('Current $(nA)$', fontsize=16)
    # pl.legend(fontsize=16)
    # pl.show()

    # downsample t, i, v
    i_inj = i_exp
    i_inj_ds = i_inj[to_idx(onset_dur, dt, 3):to_idx(tstop - offset_dur, dt, 3)]
    v_ds = v[to_idx(onset_dur, dt, 3):to_idx(tstop - offset_dur, dt, 3)]

    # compute impedance
    imp, frequencies = impedance(v_ds, i_inj_ds, dt / 1000, [freq0, freq1])  # dt in (sec) for fft

    # smooth impedance
    imp_smooth = np.array(sm.nonparametric.lowess(imp, frequencies, frac=0.3)[:, 1])

    # pl.figure()
    # pl.plot(frequencies, imp, 'k')
    # pl.plot(frequencies, imp_smooth, 'r')
    # pl.show()

    # resonance frequency
    res_freq_idx = np.argmax(imp_smooth)
    res_freq = frequencies[res_freq_idx]
    print 'resonance frequency: ' + str(res_freq)

    # use same v_rest
    v_rest = -75
    v = set_v_rest(v, v[0], v_rest)

    fig, ax1 = pl.subplots()
    ax2 = ax1.twinx().twiny()  # need two twins for labeling new x and y axis
    ax3 = ax1.twiny().twinx()
    ax1.plot((t - onset_dur) / 1000, v, 'k')
    # ylim = ax1.get_ylim()
    # ax1.set_ylim(ylim[0]-2, ylim[1]+2)
    ax1.set_ylim(v_rest - 10, v_rest + 10)
    ax1.set_xlim(0, (tstop - offset_dur - onset_dur) / 1000)
    ax2.plot(frequencies, imp_smooth, c='r', label='Res. Freq.: %.2f (Hz)' % res_freq)
    ax3.plot(frequencies, imp_smooth, c='r')
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax2.set_xlim(freq0, freq1)
    ax3.set_xlim(freq0, freq1)
    # ylim = ax3.get_ylim()
    ylim = [0, 100]
    ax3.set_ylim(ylim[0] - 5, ylim[1] + 5)
    ax2.set_ylim(ylim[0] - 5, ylim[1] + 5)
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)
    ax2.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Impedance (M$\Omega$)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Membrane Potential (mV)')

    # ax2.annotate('%.2f (Hz)' % res_freq, xy=(res_freq, imp_smooth[res_freq_idx]+0.3),
    #              xytext=(res_freq+0.5, imp_smooth[res_freq_idx]+3), fontsize=16,
    #             arrowprops=dict(arrowstyle='wedge', color='r'))
    leg = ax2.legend(handlelength=0, handletextpad=0, fancybox=True, fontsize=16)
    for item in leg.legendHandles:
        item.set_visible(False)

    pl.tight_layout()
    pl.subplots_adjust(left=0.18, right=0.86, bottom=0.14, top=0.88)
    pl.savefig(os.path.join(save_dir_img, 'v_impedance.png'))
    pl.show()


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/6'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    cell_id = '2015_08_26b'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # apply stim
    apply_zap_stimulus(cell, amp=0.1, freq0=0, freq1=20, onset_dur=2000, offset_dur=2000, zap_dur=30000, tstop=34000,
                       dt=0.1)
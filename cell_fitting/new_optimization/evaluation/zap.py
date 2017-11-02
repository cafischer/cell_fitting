from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
pl.style.use('paper')
import os
from cell_fitting.optimization.simulate import iclamp_handling_onset
from nrn_wrapper import Cell
from cell_fitting.read_heka import get_v_and_t_from_heka
from cell_fitting.data import set_v_rest


def get_zap(amp, freq0=0, freq1=20, onset_dur=2000, offset_dur=2000, tstop=34000, dt=0.01):
    """
    """
    t = np.arange(0, tstop-onset_dur-offset_dur+dt, dt)
    zap = amp * np.sin(2 * np.pi * ((freq1 - freq0) / 1000 * t / (2 * t[-1]) + freq0/1000) * t)
    onset = np.zeros(int(round(onset_dur/dt)))
    offset = np.zeros(int(round(offset_dur/dt)))
    zap_stim = np.concatenate((onset, zap, offset))

    # pl.figure()
    # pl.plot(t, zap, 'k')
    # pl.ylabel('Current (nA)')
    # pl.xlabel('Time (ms)')
    # pl.tight_layout()
    # pl.show()
    return zap_stim


def apply_zap_stimulus(cell, amp=0.1, freq0=1, freq1=20, onset_dur=2000, offset_dur=2000, tstop=34000, dt=0.01):

    i_exp = get_zap(amp, freq0, freq1, onset_dur, offset_dur, tstop, dt)

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


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/4'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    cell_id = '2015_08_26b'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # apply stim
    apply_zap_stimulus(cell, amp=0.1, freq0=0, freq1=20, onset_dur=2000, offset_dur=2000, tstop=34000, dt=0.1)  # freq0=0 or 1???  freq1=10 oder 20???
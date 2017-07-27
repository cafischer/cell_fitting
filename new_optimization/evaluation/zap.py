from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from optimization.simulate import iclamp_handling_onset
from nrn_wrapper import Cell


def get_zap(amp, freq0=1, freq1=20, onset_dur=2000, offset_dur=2000, tstop=34000, dt=0.01):
    """
    """
    t = np.arange(0, tstop-onset_dur-offset_dur+dt, dt)
    freqs = np.linspace(freq0, freq1, len(t)) / 1000
    zap = amp * np.sin(2 * np.pi * freqs * t)
    onset = np.zeros(int(round(onset_dur/dt)))
    offset = np.zeros(int(round(offset_dur/dt)))
    zap_stim = np.concatenate((onset, zap, offset))
    freqs = np.concatenate((onset, freqs, offset))
    return zap_stim


def apply_zap_stimulus(cell, amp, freq0=1, freq1=20, onset_dur=2000, offset_dur=2000, tstop=34000, dt=0.01):

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

    fig, ax1 = pl.subplots()
    ax2 = ax1.twiny()
    ax1.plot(t, v, 'r')
    #pl.plot(t, i_exp)
    ax1.set_xlim(0, t[-1])
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xticklabels(map(freqs_out, ax1.get_xticks()))
    ax2.set_xlabel('Frequency $(Hz)$', fontsize=16)
    ax1.set_xlabel('Time $(ms)$', fontsize=16)
    ax1.set_ylabel('Membrane potential $(mV)$', fontsize=16)
    pl.savefig(os.path.join(save_dir_img, 'zap.png'))
    pl.show()


if __name__ == '__main__':
    # parameters
    #save_dir = '../../results/server/2017-07-06_13:50:52/434/L-BFGS-B/'
    #model_dir = os.path.join(save_dir, 'model', 'cell.json')
    save_dir = '../../results/hand_tuning/cell_2017-07-24_13:59:54_21_0'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # apply stim
    apply_zap_stimulus(cell, amp=0.1, freq0=1, freq1=20, onset_dur=2000, offset_dur=2000, tstop=34000, dt=0.01)
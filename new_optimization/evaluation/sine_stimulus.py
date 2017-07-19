from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from optimization.simulate import iclamp_handling_onset
from nrn_wrapper import Cell


def get_sine_stimulus(amp1, amp2, sine1_dur, freq2, dt):
    """
    im Laborbuch: amp1, amp2, freq2, stim_dur
    :param amp1: amplitude of underlying sine in nA
    :param amp2: amplitude of modulating sine in nA
    :param freq2: in Hz
    :param sine1_dur: duration of big sine in ms
    :return: sine stimulus
    """
    freq2 = freq2 / 1000  # per ms
    onset_dur = 500
    offset_dur = 500
    freq1 = 1 / (sine1_dur) / 2  # per ms
    onset = np.zeros(int(round(onset_dur/dt)))
    offset = np.zeros(int(round(offset_dur/dt)))
    x = np.arange(0, sine1_dur + dt, dt)
    sine1 = np.sin(2 * np.pi * x * freq1)
    sine2 = np.sin(2 * np.pi * x * freq2)
    sine_sum = amp1*sine1 + amp2*sine2
    sine_stim = np.concatenate((onset, sine_sum, offset))
    return sine_stim


def apply_sine_stimulus(cell, amp1, amp2, sine1_dur, freq2, dt):

    i_exp = get_sine_stimulus(amp1, amp2, sine1_dur, freq2, dt)

    # get simulation parameters
    simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': -75, 'tstop': sine1_dur+1000,
                         'dt': dt, 'celsius': 35, 'onset': 200}

    # record v
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    # plot
    save_dir_img = os.path.join(save_dir, 'img', 'sine_stimulus')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    pl.figure()
    pl.title('amp1: ' + str(amp1) + ', amp2: ' + str(amp2) + ', sine1dur: ' + str(sine1_dur) + ', freq2: ' + str(freq2))
    pl.plot(t, v, 'r')
    #pl.plot(t, i_exp)
    pl.xlabel('Time $(ms)$', fontsize=16)
    pl.ylabel('Membrane potential $(mV)$', fontsize=16)
    pl.savefig(os.path.join(save_dir_img, 'sine'+str(amp1)+'_'+str(amp2)+'_'+str(sine1_dur)+'_'+str(freq2)+'_'+'.png'))
    pl.show()


if __name__ == '__main__':
    # parameters
    save_dir = '../../results/server/2017-07-06_13:50:52/434/L-BFGS-B/'
    model_dir = os.path.join(save_dir, 'model', 'cell.json')
    #save_dir = '../../results/hand_tuning/cell434_5/'
    #model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # apply stim
    amp1 = 0.6
    amp2 = 0.2
    sine1_dur = 5000
    freq2 = 5
    dt = 0.01
    apply_sine_stimulus(cell, amp1, amp2, sine1_dur, freq2, dt)

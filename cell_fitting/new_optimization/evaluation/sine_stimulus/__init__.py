from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from cell_fitting.optimization.simulate import iclamp_handling_onset
pl.style.use('paper')


def get_sine_stimulus(amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt):
    """
    im Laborbuch: amp1, amp2, freq2, stim_dur
    :param amp1: amplitude of underlying sine in nA
    :param amp2: amplitude of modulating sine in nA
    :param freq2: in Hz
    :param sine1_dur: duration of big sine in ms
    :return: sine stimulus
    """
    freq2 = freq2 / 1000  # per ms
    freq1 = 1 / (sine1_dur) / 2  # per ms
    onset = np.zeros(int(round(onset_dur/dt)))
    offset = np.zeros(int(round(offset_dur/dt)))
    x = np.arange(0, sine1_dur + dt, dt)
    sine1 = np.sin(2 * np.pi * x * freq1)
    sine2 = np.sin(2 * np.pi * x * freq2)
    sine_sum = amp1*sine1 + amp2*sine2
    sine_stim = np.concatenate((onset, sine_sum, offset))
    return sine_stim


def apply_sine_stimulus(cell, amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt):

    i_exp = get_sine_stimulus(amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt)

    # get simulation parameters
    simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': -75, 'tstop': sine1_dur+1000,
                         'dt': dt, 'celsius': 35, 'onset': 200}

    # record v
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    return v, t, i_exp
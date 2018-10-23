from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
from cell_fitting.optimization.simulate import iclamp_handling_onset
import os
pl.style.use('paper')


def evaluate_sine_stimulus(pdf, cell, save_dir):
    save_dir_img = os.path.join(save_dir, 'img', 'sine_stimulus')
    amp1 = 0.6
    amp2 = 0.2
    sine1_dur = 5000
    freq2 = 5
    onset_dur = 500
    offset_dur = 500
    dt = 0.01

    # simulate / load
    v, t, i_inj = simulate_sine_stimulus(cell, amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt)

    # plot in pdf
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)
    fig = plot_sine_stimulus(amp1, amp2, freq2, sine1_dur, t, v, save_dir_img)
    pdf.savefig(fig)
    pl.close()


def plot_sine_stimulus(amp1, amp2, freq2, sine1_dur, t, v, save_dir_img=None):
    fig = pl.figure()
    pl.plot(t, v, 'r')
    pl.ylim(-90, 60)
    pl.title('freq1: %.2f, freq2: %.2f, amp1: %.2f, amp2: %.2f' % (1./(2*sine1_dur/1000), freq2, amp1, amp2),
             fontsize=14)
    pl.ylabel('Membrane Potential (mV)')
    pl.xlabel('Time (ms)')
    pl.tight_layout()
    if save_dir_img is not None:
        pl.savefig(os.path.join(save_dir_img, 'v.png'))
    return fig


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


def simulate_sine_stimulus(cell, amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt, shift=0, v_init=-75,
                           celsius=35, onset=200, pos_v=0.5, pos_i=0.5, sec=('soma', None)):

    i_exp = get_sine_stimulus(amp1, amp2, sine1_dur, freq2, onset_dur, offset_dur, dt) + shift

    # get simulation parameters
    simulation_params = {'sec': sec, 'i_inj': i_exp, 'v_init': v_init, 'tstop': sine1_dur+1000,
                         'dt': dt, 'celsius': celsius, 'onset': onset, 'pos_v': pos_v, 'pos_i': pos_i}

    # record v
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    return v, t, i_exp
from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from cell_characteristics.analyze_APs import get_AP_onsets


def get_fft(y, dt):
    fft_y = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), d=dt)  # dt in sec

    # sort everything according to the frequencies
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    fft_y = fft_y[idx]
    return fft_y, freqs


def get_phases_fft(x, freq, window_len, dt):
    """computes the phase of the lfp corresponding to the APs for the given freq
    Input:
    window_len length of the whole window as idx"""

    dt /= 1000  # sec
    #taper = np.hanning(window_len)  # hanning window
    phases = np.zeros(len(x)-window_len)
    amp = np.zeros(len(x)-window_len)

    for i, x_i in enumerate(x[:-window_len]):

        # extract window around the x_i and taper
        x_window = x[i:i + window_len]  #* taper
        #pl.figure()
        #pl.plot(taper)
        #pl.show()

        # fft
        x_fft, freqs = get_fft(x_window, dt)

        # check that wanted frequency is present
        freqs = np.round(freqs, 2) # TODO
        if not np.any(freqs == freq):
            raise ValueError('Wanted frequency is not contained in the fft transformed signal.')

        # extract the phase
        phases[i] = np.angle(x_fft)[freqs == freq]
        amp[i] = np.abs(x_fft)[freqs == freq]

    return phases, amp


if __name__ == '__main__':
    save_dir = './results/test0/spike_theta_phase'
    save_dir_data = './results/test0/data'
    save_dir_theta = './results/test0/ramp_and_theta'

    # load
    v = np.load(os.path.join(save_dir_data, 'v.npy'))
    t = np.load(os.path.join(save_dir_data, 't.npy'))
    dt = t[1] - t[0]
    theta = np.load(os.path.join(save_dir_theta, 'theta.npy'))

    # downsampling
    for i in range(7):
        v = v[::2]
        t = t[::2]
        dt = t[1] - t[0]
        print dt # 0.001
        theta = theta[::2]

    # get phase
    phase_theta, amp = get_phases_fft(theta, freq=8, window_len=1000-219, dt=dt)
    #AP_onsets = get_AP_onsets(v, threshold=-30)
    #phase_onsets = phase_theta[AP_onsets]

    # save and plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pl.figure()
    pl.plot(t, theta, 'k')
    pl.plot(t[:len(phase_theta)], phase_theta, 'b')
    pl.plot(t[:len(amp)], amp * np.sin(2*np.pi*8*t[:len(amp)]/1000 + phase_theta) / np.max(amp), 'y')
    #pl.plot(t[AP_onsets], phase_onsets, 'or')
    pl.show()
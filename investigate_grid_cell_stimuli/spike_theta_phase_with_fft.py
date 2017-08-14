from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from grid_cell_stimuli import get_phase_and_amplitude_fft


if __name__ == '__main__':
    save_dir = './results/test0/spike_theta_phase'
    save_dir_data = './results/test0/data'
    save_dir_theta = './results/test0/ramp_and_theta'

    # params
    freq = 8
    freq2 = 5
    freq3 = 11

    # load
    v = np.load(os.path.join(save_dir_data, 'v.npy'))
    t = np.load(os.path.join(save_dir_data, 't.npy'))
    dt = t[1] - t[0]
    dt_sec = dt / 1000
    theta = np.load(os.path.join(save_dir_theta, 'theta.npy'))

    # downsampling
    for i in range(7):
        v = v[::2]
        t = t[::2]
        dt = t[1] - t[0]
        dt_sec = dt / 1000
        theta = theta[::2]

    # get phase
    phase_theta, amp_theta, freqs = get_phase_and_amplitude_fft(theta, dt=dt_sec)
    freq_idx = np.argmin(np.abs(freqs - freq))
    assert np.abs(freqs[freq_idx] - freq) < 1
    freq2_idx = np.argmin(np.abs(freqs - freq2))
    assert np.abs(freqs[freq2_idx] - freq2) < 1
    freq3_idx = np.argmin(np.abs(freqs - freq3))
    assert np.abs(freqs[freq3_idx] - freq3) < 1

    # save and plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pl.figure()
    pl.plot(t, theta, 'k')
    pl.plot(t, amp_theta[:, freq_idx] * phase_theta[:, freq_idx] / np.max(amp_theta[:, freq_idx] * phase_theta[:, freq_idx]), 'b')
    pl.plot(t, amp_theta[:, freq2_idx] * phase_theta[:, freq2_idx] / np.max(amp_theta[:, freq2_idx] * phase_theta[:, freq2_idx]), 'g')
    pl.plot(t, amp_theta[:, freq3_idx] * phase_theta[:, freq3_idx] / np.max(amp_theta[:, freq3_idx] * phase_theta[:, freq3_idx]), 'y')
    pl.show()
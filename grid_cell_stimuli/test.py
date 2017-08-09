from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from scipy.signal import spectrogram
from grid_cell_stimuli import get_phase_and_amplitude_fft


if __name__ == '__main__':
    save_dir = './results/test0/spike_theta_phase'
    save_dir_data = './results/test0/data'
    save_dir_theta = './results/test0/ramp_and_theta'

    # load
    freq1 = 8
    freq2 = 1
    dt = 1  # ms
    dt_sec = dt / 1000
    t = np.arange(0, 5000+dt, dt)  # ms
    x = np.concatenate((np.sin(2 * np.pi * freq1 * t[:int(round(len(t) / 2))] / 1000),
                        np.sin(2 * np.pi * freq2 * t[int(round(len(t)/2)):] / 1000 + np.pi)))

    # get phase
    phase_fft, amp_fft, freqs = get_phase_and_amplitude_fft(x, dt_sec)
    freq_idx = np.argmin(np.abs(freqs - freq1))
    assert np.abs(freqs[freq_idx] - freq1) < 1
    freq2_idx = np.argmin(np.abs(freqs - freq2))
    assert np.abs(freqs[freq2_idx] - freq2) < 1

    phase1 = phase_fft[:, freq_idx]
    phase2 = phase_fft[:, freq2_idx]
    amp1 = amp_fft[:, freq_idx]
    amp2 = amp_fft[:, freq2_idx]

    pl.figure()
    pl.plot(t, x, 'k', alpha=0.5, linewidth=2)
    #pl.plot(t, amp1 * phase1 / np.max(amp1 * phase1), 'b', alpha=0.5, linewidth=2)
    #pl.plot(t, amp2 * phase2 / np.max(amp2 * phase2), 'g', alpha=0.5, linewidth=2)
    pl.plot(t, amp1 * phase1, 'b', alpha=0.5, linewidth=2)
    pl.plot(t, amp2 * phase2, 'g', alpha=0.5, linewidth=2)
    pl.show()

    # # get phase by spectogram
    # freq_spec, t_spec, phase_spec = spectrogram(x, fs=1 / dt_sec, mode='angle', nperseg=1, nfft=5)
    # freq_spec, t_spec, pow_spec = spectrogram(x, fs=1 / dt_sec, mode='magnitude', scaling='spectrum', nperseg=1, nfft=5)
    # print t_spec
    #
    # pl.figure()
    # pl.pcolormesh(t_spec, freq_spec, pow_spec)
    # pl.show()
    #
    # pl.figure()
    # pl.pcolormesh(t_spec, freq_spec, phase_spec)
    # pl.show()
    #
    # theta_freq = (6, 11)
    # phases_theta = phase_spec[np.logical_and(freq_spec > theta_freq[0], freq_spec < theta_freq[1]), :]
    #
    # pl.figure()
    # pl.plot(t, x, 'k')
    # pl.plot(t_spec*1000, np.mean(phases_theta, 0), 'b')
    # pl.show()
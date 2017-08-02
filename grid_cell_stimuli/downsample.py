import matplotlib.pyplot as pl
import numpy as np
import os
import copy
from scipy.signal import firwin, freqz, kaiserord


def get_fft(y, dt):
    fft_y = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), d=dt)  # dt in sec

    # sort everything according to the frequencies
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    fft_y = fft_y[idx]
    return fft_y, freqs


if __name__ == '__main__':

    save_dir = './results/test0/downsampled'
    save_dir_data = './results/test0/APs_removed'

    # parameters
    dt_new_max = 0.1  # ms (=10 kHz)
    cutoff_freq = 5000  # Hz
    width = 5.0  # Hz
    ripple_attenuation = 60.0  # db

    # load
    v = np.load(os.path.join(save_dir_data, 'v.npy'))
    t = np.load(os.path.join(save_dir_data, 't.npy'))
    dt = t[1] - t[0]

    # downsample
    dt_sec = dt / 1000
    sample_rate = 1.0 / dt_sec
    nyq_rate = sample_rate / 2.0
    N, beta = kaiserord(ripple_attenuation, width / nyq_rate)
    assert N < len(v)  # filter not bigger than data to filter
    filter_downsample = firwin(N+1, cutoff_freq / nyq_rate, window=('kaiser', beta), pass_zero=True)  # pass_zeros True
                                                                                                      # for low-pass

    v_antialiased = np.convolve(v, filter_downsample, mode='valid')
    idx_cut = int(np.ceil((len(t) - len(v_antialiased)) / 2.0))
    t_antialiased = np.round(t[idx_cut: -idx_cut] - t[idx_cut], 5)

    downsample_rate = 2
    n = np.floor(np.log(dt_new_max / dt) / np.log(downsample_rate))
    v_downsampled = copy.copy(v_antialiased)
    t_downsampled = copy.copy(t_antialiased)
    for i in range(3):
        v_downsampled = v_downsampled[::downsample_rate]
        t_downsampled = t_downsampled[::downsample_rate]

    # save and plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'v.npy'), v_downsampled)
    np.save(os.path.join(save_dir, 't.npy'), t_downsampled)

    pl.figure()
    pl.plot(t, v, 'b', label='$V_{APs\ removed}$')
    pl.plot(t_downsampled + t[idx_cut], v_downsampled, 'g', label='$V_{downsampled}$')
    pl.ylabel('Membrane potential (mV)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'v_downsampled.png'))
    pl.show()

    pl.figure()
    w, h = freqz(filter_downsample, worN=int(round(nyq_rate / 0.1)))
    pl.plot((w / np.pi) * nyq_rate, np.absolute(h))
    pl.xlabel('Frequency (Hz)',fontsize=16)
    pl.ylabel('Gain', fontsize=16)
    pl.ylim(-0.05, 1.05)
    pl.xlim(0, 10000)
    pl.savefig(os.path.join(save_dir, 'gain_filter.png'))
    pl.show()

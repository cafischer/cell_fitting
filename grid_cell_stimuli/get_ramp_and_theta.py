import matplotlib.pyplot as pl
import numpy as np
import os
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

    save_dir = './results/test0/ramp_and_theta'
    save_dir_data = './results/test0/downsampled'
    # save_dir_data = './results/test0/APs_removed'

    # parameters
    cutoff_ramp = 3  # Hz
    cutoff_theta_low = 5  # Hz
    cutoff_theta_high = 11  # Hz
    width = 1  # Hz
    ripple_attenuation = 60.0  # db

    # load
    v = np.load(os.path.join(save_dir_data, 'v.npy'))
    t = np.load(os.path.join(save_dir_data, 't.npy'))
    dt = t[1] - t[0]

    # get ramp and theta
    dt_sec = dt / 1000
    sample_rate = 1.0 / dt_sec
    nyq_rate = sample_rate / 2.0
    N, beta = kaiserord(ripple_attenuation, width / nyq_rate)
    assert N < len(v)  # filter not bigger than data to filter
    filter_ramp = firwin(N+1, cutoff_ramp / nyq_rate, window=('kaiser', beta), pass_zero=True)
    filter_theta = firwin(N+1, [cutoff_theta_low / nyq_rate, cutoff_theta_high / nyq_rate], window=('kaiser', beta),
                          pass_zero=False)  # pass_zero seems to flip from bandpass to bandstop

    ramp = np.convolve(v, filter_ramp, mode='valid')
    theta = np.convolve(v, filter_theta, mode='valid')
    idx_cut = int(np.ceil((len(t) - len(ramp)) / 2.0))
    t_filtered = np.arange(0, len(ramp) * dt, dt)

    # save and plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'ramp.npy'), ramp)
    np.save(os.path.join(save_dir, 'theta.npy'), theta)
    np.save(os.path.join(save_dir, 't.npy'), t_filtered)

    pl.figure()
    w, h = freqz(filter_ramp, worN=int(round(nyq_rate / 0.01)))
    pl.plot((w / np.pi) * nyq_rate, np.absolute(h), label='Ramp')
    w, h = freqz(filter_theta, worN=int(round(nyq_rate / 0.01)))
    pl.plot((w / np.pi) * nyq_rate, np.absolute(h), label='Theta')
    pl.xlabel('Frequency (Hz)', fontsize=16)
    pl.ylabel('Gain', fontsize=16)
    pl.ylim(-0.05, 1.05)
    pl.xlim(0, 20)
    pl.legend(fontsize=16)
    pl.show()

    pl.figure()
    v_fft, freqs = get_fft(v, dt_sec)
    pl.plot(freqs, np.abs(v_fft) ** 2, 'k', label='V')
    ramp_fft, freqs = get_fft(ramp, dt_sec)
    pl.plot(freqs, np.abs(ramp_fft) ** 2, 'g', label='Ramp')
    theta_fft, freqs = get_fft(theta, dt_sec)
    pl.plot(freqs, np.abs(theta_fft) ** 2, 'b', label='Theta')
    pl.xlabel('Frequency', fontsize=16)
    pl.ylabel('Power', fontsize=16)
    pl.xlim(0, 50)
    pl.ylim(0, 1e9)
    pl.legend(fontsize=16)
    pl.show()

    pl.figure()
    pl.plot(t, v, 'k')
    pl.plot(t_filtered + t[idx_cut], ramp, 'g', linewidth=2, label='Ramp')
    pl.plot(t_filtered + t[idx_cut], theta - 75, 'b', linewidth=2, label='Theta')
    pl.xlabel('t')
    pl.xlim(t_filtered[0] + t[idx_cut], t_filtered[-1] + t[idx_cut])
    pl.ylabel('Voltage (mV)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.legend(fontsize=16)
    pl.show()
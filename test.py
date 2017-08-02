import numpy as np
import matplotlib.pyplot as pl
from scipy.signal import firwin, freqz, kaiserord


def get_fft(y, dt):
    fft_y = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), d=dt)  # dt in sec

    # sort everything according to the frequencies
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    fft_y = fft_y[idx]
    return fft_y, freqs


# test signal
dt = 0.01 #/ 1000  # sec
t = np.arange(0, 10, dt)
signal = (np.sin(2 * np.pi * 2 * t + 1) + np.sin(2 * np.pi * 4 * t + 2) +
          + np.sin(2 * np.pi * 5 * t + 3))

# FIR filter
sample_rate = 1.0 / dt
nyq_rate = sample_rate / 2.0
width = 1.0/nyq_rate
ripple_db = 60.0
N, beta = kaiserord(ripple_db, width)
if N % 2 == 0:
    N += 1
cutoff_low = 0.00001  # Hz
cutoff_high = 3  # Hz
filter_ramp = firwin(N, [cutoff_low / nyq_rate, cutoff_high / nyq_rate], window=('kaiser', beta), pass_zero=False)

cutoff_low = 5  # Hz
cutoff_high = 12  # Hz
filter_theta = firwin(N, [cutoff_low / nyq_rate, cutoff_high / nyq_rate], window=('kaiser', beta), pass_zero=False)

filtered_signal = np.convolve(signal, filter_ramp, mode='same')  # , mode='valid')


#plots
pl.figure()
pl.xlabel('Frequency')
pl.ylabel('Power')
signal_fft, freqs = get_fft(signal, dt)
pl.plot(freqs, np.abs(signal_fft)**2, 'k', linewidth=3)
filtered_x_fft, freqs = get_fft(filtered_signal, dt)
pl.plot(freqs, np.abs(filtered_x_fft)**2, 'b')
pl.xlim(0, 10)
pl.show()

# plots
pl.figure()
pl.plot(filter_ramp, 'bo-', linewidth=2)
pl.title('Filter (length: %d)' % N)

pl.figure()
w, h = freqz(filter_ramp, worN=1000)
pl.plot((w / np.pi) * nyq_rate, np.absolute(h), linewidth=2)
pl.xlabel('Frequency (Hz)')
pl.ylabel('Gain')
pl.ylim(-0.05, 1.05)
pl.xlim(0, 20)

pl.figure()
pl.plot(t, signal)
pl.plot(t, filtered_signal, 'g', linewidth=2)
pl.xlabel('t')
pl.show()


# a = 10
# b = 0.2
# c = 0.004
# d = 0.8
#
# x = np.arange(-100, 40)
# y1 = a * np.exp(-b*x)
# y2 = c * np.exp(d*x)
# inf = y1 / (y1 + y2)
# tau = 1 / (y1 + y2)
#
# pl.figure()
# pl.plot(x, y1, 'b', label='alpha')
# pl.plot(x, y2, 'g', label='beta')
# pl.plot(x, inf, 'r', label='inf')
# pl.plot(x, tau, 'y', label='tau')
# pl.ylim(0, 15)
# pl.legend()
# pl.show()

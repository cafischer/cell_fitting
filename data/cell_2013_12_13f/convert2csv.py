from __future__ import division
import numpy as np
import pylab as pl
import csv
import statsmodels.api as sm

__author__ = 'caro'

def impedance(v, i, dt, f_range):
    # FFT of the membrance potential and the input current
    fft_i = np.fft.fft(i)
    fft_v = np.fft.fft(v)
    freqs = np.fft.fftfreq(v.size, d=dt)

    # sort everything according to the frequencies
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    fft_i = fft_i[idx]
    fft_v = fft_v[idx]

    # calculate the impedance
    imp = np.abs(fft_v/fft_i)

    # index with frequency range
    idx1 = np.argmin(np.abs(freqs-f_range[0]))
    idx2 = np.argmin(np.abs(freqs-f_range[1]))

    return imp[idx1:idx2], freqs[idx1:idx2]

def phase_shift(v, i, dt, f_range):
    # FFT of the membrance potential and the input current
    fft_i = np.fft.fft(i)
    fft_v = np.fft.fft(v)
    freqs = np.fft.fftfreq(v.size, d=dt)

    # sort everything according to the frequencies
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    fft_i = fft_i[idx]
    fft_v = fft_v[idx]

    # calculate the phase shift between the somatic membrane potential and the input current
    phase_shift = np.angle(fft_v)-np.angle(fft_i)

    # index with frequency range
    idx1 = np.argmin(np.abs(freqs-f_range[0]))
    idx2 = np.argmin(np.abs(freqs-f_range[1]))

    return phase_shift[idx1:idx2], freqs[idx1:idx2]


# load data
v = np.array(np.loadtxt('./zap_current/Zap_voltage.txt', delimiter="\n"))
i = np.array(np.loadtxt('./zap_current/Zap_current.txt', delimiter="\n"))
t = np.array(np.loadtxt('./zap_current/Zap_time.txt', delimiter="\n"))

"""
# plot
pl.figure()
pl.plot(t, v)
pl.show()
pl.figure()
pl.plot(t, i)
pl.show()


# save as csv file
header = np.array(['t', 'v', 'i', 'sec'], dtype=str)
sec = np.zeros(len(t), dtype=object)
sec[0] = 'soma'
data_new = np.column_stack((t, v/1000, i, sec))
with open('./zap_current/zap.csv', 'w') as csvoutput:
    writer = csv.writer(csvoutput, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(data_new)
"""

# compute impedance
t_tmp = t * 1000  # convert to ms
f_range = [1,19]
imp, freqs = impedance(v, i, t_tmp[1]-t_tmp[0], f_range)

"""
import scipy.signal

imp_smooth = scipy.signal.savgol_filter(imp, 51, 3)

pl.figure()
pl.plot(freqs, imp_smooth)
pl.plot(freqs, imp)
pl.show()
"""

"""
import scipy.fftpack

w = scipy.fftpack.rfft(imp)
f = scipy.fftpack.rfftfreq(len(freqs), freqs[1]-freqs[0])
spectrum = w**2

cutoff_idx = spectrum < (spectrum.max()/5)
w2 = w.copy()
w2[cutoff_idx] = 0

y2 = scipy.fftpack.irfft(w2)

pl.figure()
pl.plot(freqs, y2)
pl.plot(freqs, imp)
pl.show()
"""

# smooth impedance
imp_smooth = sm.nonparametric.lowess(imp, freqs, frac=0.5)
imp_smooth = np.array(imp_smooth[:, 1])

# plot
#pl.figure()
#pl.plot(freqs, imp_smooth)
#pl.plot(freqs, imp)
#pl.show()

# save as csv file
header = np.array(['t', 'i', 'v', 'impedance', 'f_range', 'sec'], dtype=str)
sec = np.zeros(np.array(t).size, dtype=object)
sec[0] = 'soma'
f_range2 = np.zeros(np.array(t).size)
f_range2[0] = f_range[0]
f_range2[1] = f_range[1]
imp_smooth2 = np.zeros(np.array(t).size, dtype=object)
imp_smooth2[:] = np.NAN
imp_smooth2[:imp_smooth.size] = imp_smooth
data = np.column_stack((t, i, v/1000, imp_smooth2, f_range2, sec))
with open('./zap_current/impedance.csv', 'w') as csvoutput:
    writer = csv.writer(csvoutput, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(data)

"""
# compute phase shift
dt = np.diff(t)[0]
phase_shift, freqs = phase_shift(v,i,dt,[1,19])

# plot
pl.figure()
pl.plot(freqs, phase_shift)
pl.show()
"""
from __future__ import division
import numpy as np
import pylab as pl
import csv
import statsmodels.api as sm

__author__ = 'caro'


def impedance(v, i, dt, f_range):
    """
    Computes the impedance (impedance = fft(v) / fft(i)) for a given range of frequencies.

    :param v: Membrane potential (mV)
    :type v: array
    :param i: Current (nA)
    :type i: array
    :param dt: Time step.
    :type dt: float
    :param f_range: Boundaries of the frequency interval.
    :type f_range: list
    :return: Impedance (MOhm)
    :rtype: array
    """

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


def zap_current2csv(cell_dir):
    """
    Reads the data for the zap experiment and writes it to a csv file. Furthermore computes the impedance and saves it
    (together with the other information ) to a csv file.
    """

    # load data
    f_range = [1, 20]
    ds = 1000  # number of steps skipped (in t, i, v) for the impedance computation
    v = np.array(np.loadtxt(cell_dir + '/zapcurrent/zap_voltage.txt', delimiter='\n'))
    i = np.array(np.loadtxt(cell_dir + '/zapcurrent/zap_current.txt', delimiter="\n"))
    t = np.array(np.loadtxt(cell_dir + '/zapcurrent/zap_time.txt', delimiter="\n"))

    # convert units
    t *= 1000  # (ms)

    # plot data
    f, (ax1, ax2) = pl.subplots(2, 1, sharex=True)
    ax1.plot(t, v)
    ax1.set_ylabel('Membrane potential (mV)')
    ax2.plot(t, i)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Current (nA)')
    pl.show()

    # save to .csv
    header = ['t', 'i', 'v', 'sec']  # units: (ms), (nA), (mV)
    sec = np.zeros(len(t), dtype=object)  # section at which was recorded
    sec[0] = 'soma'
    data_new = np.column_stack((t, i, v, sec))
    with open(cell_dir + '/zapcurrent/zap.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(data_new)

    # downsample t, i, v
    t = t[::ds]
    i = i[::ds]
    v = v[::ds]

    # compute impedance
    imp, freqs = impedance(v, i, t[1]/1000-t[0]/1000, f_range)  # dt in (sec) for fft

    # smooth impedance
    imp_smooth = np.array(sm.nonparametric.lowess(imp, freqs, frac=0.5)[:, 1])

    # plot impedance
    pl.figure()
    pl.plot(freqs, imp, label='impedance')
    pl.plot(freqs, imp_smooth, label='smoothed impedance', color='r')
    pl.show()

    # save to .csv
    header = ['t', 'i', 'v', 'impedance', 'f_range', 'sec']  # units: (ms), (nA), (mV), (MOhm), (Hz)
    sec = np.zeros(np.array(t).size, dtype=object)
    sec[0] = 'soma'
    f_range2save = np.zeros(np.array(t).size)
    f_range2save[0] = f_range[0]
    f_range2save[1] = f_range[1]
    imp_smooth2save = np.zeros(np.array(t).size, dtype=object)
    imp_smooth2save[:] = np.NAN
    imp_smooth2save[:imp_smooth.size] = imp_smooth
    data = np.column_stack((t, i, v, imp_smooth2save, f_range2save, sec))
    with open(cell_dir + '/zapcurrent/impedance_ds.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(data)

if __name__ == "__main__":
    zap_current2csv('./cell_2013_12_13f')
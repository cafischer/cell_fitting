import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from data.hekareader import *

if __name__ == '__main__':
    file_dir = '../data/2015_08_26b/2015_08_26b.dat'
    hekareader = HekaReader(file_dir)
    protocol = 'hypTester'
    indices = get_indices_for_protocol(hekareader, protocol)

    v_traces = np.zeros((len(indices), len(hekareader.get_xy(indices[0])[0])))
    for i, index in enumerate(indices):
        t, v_traces[i, :] = hekareader.get_xy(index)
        t_unit, v_unit = hekareader.get_units_xy(index)

    # compute mean
    v_mean = np.mean(v_traces, 0)
    pl.figure()
    pl.plot(t, v_mean, 'k')
    pl.xlabel('Time (' + t_unit + ')')
    pl.ylabel('Membrane Potential (' + v_unit + ')')
    pl.show()

    # load i_inj
    i_inj = np.array(pd.read_csv('../data/Protocols/'+protocol+'.csv', header=None)[0].values)
    start_i_inj = np.where(i_inj < 0)[0][0]
    end_i_inj = np.where(i_inj[start_i_inj:] == 0)[0][0] + start_i_inj

    # fit tau (decay time constant)
    def exponential(t, tau):
        return v_diff * np.exp(-t / tau) - v_diff

    max_depolarization_idx = np.argmin(v_mean)
    v_expdecay = v_mean[start_i_inj:max_depolarization_idx] - v_mean[start_i_inj]
    t_expdecay = t[start_i_inj:max_depolarization_idx] - t[start_i_inj]
    v_diff = np.abs(v_expdecay[-1] - v_expdecay[0])
    tau = curve_fit(exponential, t_expdecay, v_expdecay)[0][0]

    pl.figure()
    pl.plot(t_expdecay, v_expdecay, 'k')
    pl.plot(t_expdecay, exponential(t_expdecay, tau), 'r')
    pl.show()

    # compute Rin
    v_rest = np.mean(v_mean[:start_i_inj - 1])
    v_in = np.min(v_mean) - v_rest
    i_in = i_inj[start_i_inj]

    r_in = np.abs(v_in / i_in) * 1000  # V / nA to MOhm

    # compute capacitance
    c_m = tau / r_in * 10 ** 6  # sec / MOhm to pF
    print 'tau: ' + str(tau * 1000) + ' ms'
    print 'R_in: ' + str(r_in) + ' MOhm'
    print 'c_m: ' + str(c_m) + ' pF'
from __future__ import division
import os
import matplotlib.pyplot as pl
import numpy as np
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.simulate import iclamp_handling_onset
from cell_characteristics import to_idx
from statsmodels import api as sm
from neuron import h


def ou_noise_input(cell, g_e0=0.01, g_i0=0.07, std_e=0.003, std_i=0.005, tau_e=2.0, tau_i=5.0, E_e=0, E_i=-75):
    ou_process = h.Gfluct(cell.soma(0.5))
    ou_process.g_e0 = g_e0  #0.00047 # 0.01  # average excitatory conductance
    ou_process.g_i0 = g_i0  #0.00392 # 0.07  # average inhibitory conductance
    ou_process.std_e = std_e  #0.00240 # 0.003  # standard deviation of excitatory conductance
    ou_process.std_i = std_i  #0.00104 # 0.005  # standard deviation of inhibitory conductance
    ou_process.tau_e = tau_e  #2.5  # 2.0  # time constant of excitatory conductance
    ou_process.tau_i = tau_i  #10.0  # 5.0  # time constant of inhibitory conductance
    ou_process.E_e = E_e  # excitatory reversal potential
    ou_process.E_i = E_i  # inhibitory reversal potential
    return ou_process


if __name__ == '__main__':

    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/5'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'
    load_mechanism_dir(mechanism_dir)

    onset = 200
    dt = 0.01
    celsius = 35
    v_init = -75
    tstop = 1000
    n_trials = 10

    # create cell
    cell = Cell.from_modeldir(model_dir)

    # synaptic input
    ou_process = ou_noise_input(cell, g_e0=0.01, g_i0=0.07, std_e=0.003, std_i=0.005, tau_e=2.0, tau_i=5.0)

    # simulate
    medians = np.zeros(n_trials)
    means = np.zeros(n_trials)
    stds = np.zeros(n_trials)
    v_traces = []
    t_traces = []
    for n in range(n_trials):
        ou_process.new_seed(n)
        simulation_params = {'sec': ('soma', None), 'i_inj': np.zeros(to_idx(tstop, dt)), 'v_init': v_init, 'tstop': tstop,
                             'dt': dt, 'celsius': celsius, 'onset': onset}
        v, t, _ = iclamp_handling_onset(cell, **simulation_params)
        medians[n] = np.median(v)
        means[n] = np.mean(v)
        stds[n] = np.std(v)
        v_traces.append(v)
        t_traces.append(t)

        # plot
        pl.figure()
        pl.plot(t, v, 'k')
        pl.ylabel('Membrane potential (mV)', fontsize=16)
        pl.xlabel('Time (ms)', fontsize=16)
        pl.tight_layout()

        pl.figure()
        pl.hist(v, bins=100)
        pl.ylabel('Count', fontsize=16)
        pl.xlabel('Membrane potential (mV)', fontsize=16)
        pl.tight_layout()
        pl.show()

    print 'Average Median: %.2f' % np.mean(medians)
    print 'Average Mean: %.2f' % np.mean(means)
    print 'Average Std: %.2f' % np.mean(stds)

    # for v, t in zip(v_traces, t_traces):
    #     auto_corr = sm.tsa.acf(v, nlags=len(v))
    #     pl.figure()
    #     pl.plot(t, auto_corr)
    #     pl.xlabel('Time lag (ms)')
    #     pl.ylabel('Autocorrelation')
    #     pl.show()

        # auto_corr = sm.tsa.acf(v, nlags=len(v))
        # auto_corr2 = sm.tsa.acf(v[::-1], nlags=len(v))
        # pl.figure()
        # pl.plot(np.concatenate((-1*t[::-1], t)), np.concatenate((auto_corr2[::-1], auto_corr)))
        # pl.xlabel('Time lag (ms)')
        # pl.ylabel('Autocorrelation')
        # pl.show()
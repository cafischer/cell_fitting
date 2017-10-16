import os
from nrn_wrapper import Cell, load_mechanism_dir
from neuron import h
from cell_fitting.new_optimization.evaluation.IV import get_step
from cell_characteristics import to_idx
from cell_fitting.optimization.simulate import iclamp_handling_onset
from grid_cell_stimuli.downsample import antialias_and_downsample
import scipy.signal
import numpy as np
import matplotlib.pyplot as pl
pl.style.use('paper')


if __name__ == '__main__':

    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/5'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../model/channels/vavoulis'
    ou_dir = '../../model/OU_process'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # insert Ornstein-Uhlenbeck process
    load_mechanism_dir(ou_dir)
    ou_process = h.Gfluct(cell.soma(0.5))
    ou_process.new_seed(5)
    ou_process.g_e0 = 0.005  # average excitatory conductance
    ou_process.g_i0 = 0.04

    # simulation_params
    tstop = 5000
    dt = 0.01
    hold_amp = 0.1
    v_init = -75
    onset = 500
    celsius = 35

    # simulate
    i_hold = get_step(0, to_idx(tstop, dt) + 1, to_idx(tstop, dt) + 1, hold_amp)
    i_exp = i_hold

    simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': v_init,
                         'tstop': tstop, 'dt': dt, 'celsius': celsius, 'onset': onset}
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    # plot
    pl.figure()
    pl.plot(t, v, 'r')
    pl.xlabel('Time (ms)')
    pl.ylabel('Membrane Potential (mV)')
    pl.tight_layout()
    # pl.show()

    # periodogram
    window_t = 2000
    window_idx = window_t / dt
    window_idx_p2 = int(np.ceil(np.sqrt(window_idx))**2)

    dt = t[1] - t[0]
    print dt
    cutoff_freq = 50  # Hz
    dt_new_max = 1. / cutoff_freq * 1000  # ms
    transition_width = 5.0  # Hz
    ripple_attenuation = 60.0  # db
    v_downsampled, t_downsampled, filter = antialias_and_downsample(v, dt, ripple_attenuation, transition_width,
                                                                    cutoff_freq, dt_new_max)
    dt = t_downsampled[1] - t_downsampled[0]
    fs = 1/dt*1000
    print dt

    #scipy.signal.periodogram(v, fs=fs, nfft=window_idx_p2)
    pl.figure()
    pl.specgram(v_downsampled, Fs=fs, NFFT=window_idx_p2)
    pl.colorbar()
    pl.tight_layout()
    pl.show()
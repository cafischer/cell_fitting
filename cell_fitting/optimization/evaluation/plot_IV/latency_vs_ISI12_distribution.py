from __future__ import division
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.optimization.evaluation.plot_IV import get_step, get_IV
import matplotlib.pyplot as pl
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_characteristics.analyze_step_current_data import get_latency_to_first_spike, get_ISI12
pl.style.use('paper')


__author__ = 'caro'


def get_latency_and_ISI12(cell):
    start_step = 250  # ms
    end_step = 750  # ms
    tstop = 1000  # ms
    step_amps = np.arange(0.0, 1.0, 0.05)
    AP_threshold = 0
    dt = 0.01

    # simulate
    v_mat = np.zeros((len(step_amps), int(tstop/dt)+1))
    for i, step_amp in enumerate(step_amps):
        v_mat[i, :], t, i_inj = get_IV(cell, step_amp, get_step, start_step, end_step, tstop, dt)

    # get latency and ISI12
    latency = None
    for v in v_mat:
        AP_onsets = get_AP_onset_idxs(v, AP_threshold)
        latency = get_latency_to_first_spike(v, t, AP_onsets, start_step, end_step)
        if latency is not None:
            break

    ISI12 = None
    for v in v_mat:
        AP_onsets = get_AP_onset_idxs(v, AP_threshold)
        ISI12 = get_ISI12(v, t, AP_onsets, start_step, end_step)
        if ISI12 is not None:
            break

    return latency, ISI12


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    latency, ISI12 = get_latency_and_ISI12(cell)



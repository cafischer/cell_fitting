from __future__ import division
import numpy as np
import os
from nrn_wrapper import Cell
from cell_fitting.optimization.evaluation.plot_IV import get_step, get_IV
import matplotlib.pyplot as pl
from cell_characteristics.analyze_APs import get_AP_onset_idxs
pl.style.use('paper')


__author__ = 'caro'


def get_latency_and_ISI12(cell):
    start_step = 250  # ms
    end_step = 750  # ms
    tstop = 1000  # ms
    step_amps = np.arange(0.0, 1.0, 0.05)
    AP_threshold = 0
    dt = 0.01

    # main
    latency_found = False
    ISI12_found = False
    for step_amp in step_amps:
        v, t, i_inj = get_IV(cell, step_amp, get_step, start_step, end_step, tstop, dt)
        AP_onsets = get_AP_onset_idxs(v, AP_threshold)

        # get latency and ISI1/2
        if latency_found == False and len(AP_onsets) >= 1:
            latency = t[AP_onsets[0]] - start_step
            latency_found = True
            # print 'Latency (ms)', latency
            # pl.figure()
            # pl.plot(t, v)
            # pl.show()

        if ISI12_found == False and len(AP_onsets) >= 4:
            ISIs = np.diff(AP_onsets) * dt
            ISI12 = ISIs[0] / ISIs[1]
            ISI12_found = True
            # print 'ISI1/2 (ms): ', ISI12
            # pl.figure()
            # pl.plot(t, v)
            # pl.show()

        if latency_found and ISI12_found:
            return latency, ISI12

    return None, None


if __name__ == '__main__':
    # parameters
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    latency, ISI12 = get_latency_and_ISI12(cell)



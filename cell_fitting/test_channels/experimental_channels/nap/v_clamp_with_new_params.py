from __future__ import division
from nrn_wrapper import Cell
import numpy as np
from test_channels.test_ionchannel import current_subtraction
import os
import pandas as pd
import matplotlib.pyplot as pl
import json


if __name__ == "__main__":

    # channel to investigate
    channel = "nap_fit"
    model_dir = '../../../model/cells/dapmodel_nocurrents.json'
    mechanism_dir = './mod/'
    save_dir = '/media/caro/Daten/Phd/DAP-Project/cell_fitting/results/ion_channels/nap_new/L-BFGS-B/best_candidate.json'
    data_dir = os.path.join('.', 'plots', 'digitized_vsteps', 'traces.csv')

    # load data
    all_traces = pd.read_csv(data_dir, index_col=0)
    all_traces /= np.max(np.max(np.abs(all_traces)))

    # parameters
    celsius = 24
    amps = [0, 0, 0]
    durs = [0, 480, 0]
    v_steps = [float(v) for v in all_traces.columns.values]
    stepamp = 2
    pos = 0.5
    dt = all_traces.index[1] - all_traces.index[0]

    # create cell
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    cell.insert_mechanisms([[['soma', '0.5', channel, 'gbar']]])
    cell.update_attr(['soma', '0.5', channel, 'gbar'], 1.0)
    sec_channel = getattr(cell.soma(.5), channel)

    # change params
    with open(save_dir, 'r') as f:
        best_candidate = json.load(f)
    for k, v in best_candidate.iteritems():
        cell.update_attr(['soma', '0.5', channel, k], v)

    # compute response to voltage steps
    i_steps, t = current_subtraction(cell.soma, sec_channel, celsius, amps, durs, v_steps, stepamp, pos, dt)

    # compare to experimental data
    scale_fac = 1.0 / np.max(np.abs(np.matrix(i_steps)[:, 1:]))
    pl.figure()
    for i, column in enumerate(all_traces.columns):
        pl.plot(all_traces.index, all_traces[column], 'k', label=column)
        pl.plot(t[:-1], i_steps[i][1:] * scale_fac, 'r')
    pl.ylabel('Current (pA)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.legend(fontsize=16)
    pl.show()
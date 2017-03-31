from __future__ import division
from nrn_wrapper import Cell
import numpy as np
from test_channels.test_ionchannel import voltage_steps
import os
import pandas as pd
import matplotlib.pyplot as pl
import json


if __name__ == "__main__":

    # channel to investigate
    channel = "hcn_fit"
    model_dir = '../../../model/cells/dapmodel_nocurrents.json'
    mechanism_dir = './mod/'
    save_dir = '/media/caro/Daten/Phd/DAP-Project/cell_fitting/results/ion_channels/hcn/L-BFGS-B/best_candidate.json'
    data_dir = os.path.join('.', 'plots', 'digitized_vsteps2', 'traces.csv')

    # load data
    all_traces = pd.read_csv(data_dir, index_col=0)
    all_traces /= np.max(np.max(np.abs(all_traces)))

    # parameters
    celsius = 24
    amps = [0, 0, 0]
    durs = [0, 1565, 0]
    v_steps = [float(v) for v in all_traces.columns.values]
    stepamp = 2
    pos = 0.5
    dt = all_traces.index[1] - all_traces.index[0]

    # create cell
    cell = Cell.from_modeldir(model_dir, mechanism_dir)

    # change params
    with open(save_dir, 'r') as f:
        best_candidate = json.load(f)
    cell.insert_mechanisms([[['soma', '0.5', channel, 'gfastbar']]])
    cell.update_attr(['soma', '0.5', channel, 'gfastbar'], best_candidate['g_m'])
    cell.insert_mechanisms([[['soma', '0.5', channel, 'gslowbar']]])
    cell.update_attr(['soma', '0.5', channel, 'gslowbar'], best_candidate['g_h'])
    sec_channel = getattr(cell.soma(.5), channel)

    del best_candidate['g_m']
    del best_candidate['g_h']
    for k, v in best_candidate.iteritems():
        cell.update_attr(['soma', '0.5', channel, k], v)

    # compute response to voltage steps
    i_steps = []
    i_steps_control, t = voltage_steps(cell.soma, amps, durs, v_steps, stepamp, pos, dt)
    setattr(sec_channel, 'gfastbar', 0.0)
    setattr(sec_channel, 'gslowbar', 0.0)
    i_steps_blockade, t = voltage_steps(cell.soma, amps, durs, v_steps, stepamp, pos, dt)

    for i, v_step in enumerate(v_steps):
        i_steps.append(i_steps_control[i] - i_steps_blockade[i])

    # compare to experimental data
    scale_fac = 1.0 / np.max(np.abs(np.matrix(i_steps)[:, 1:]))
    pl.figure()
    for i, column in enumerate(all_traces.columns):
        pl.plot(all_traces.index, all_traces[column], 'k', label=column)
        pl.plot(t[:-1], i_steps[i] * scale_fac, 'r')
    pl.ylabel('Current (pA)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.legend(fontsize=16)
    pl.show()
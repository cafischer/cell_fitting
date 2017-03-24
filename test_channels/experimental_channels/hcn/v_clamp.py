from __future__ import division
import numpy as np
from nrn_wrapper import Cell
import pandas as pd
import os
import matplotlib.pyplot as pl
from test_channels.test_ionchannel import voltage_steps, plot_i_steps


if __name__ == "__main__":

    # channel to investigate
    channel = "hcn_dickson"
    model_dir = '../../../model/cells/dapmodel_nocurrents.json'
    mechanism_dir = './mod/'

    # parameters
    amps = [0, 1, 0]
    durs = [0, 1600, 0]
    v_steps = np.linspace(-125, -55, 15)
    stepamp = 2
    pos = 0.5
    dt = 0.025

    # create cell
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    cell.insert_mechanisms([[['soma', '0.5', channel, 'gfastbar']]])
    cell.update_attr(['soma', '0.5', channel, 'gfastbar'], 1.0)
    cell.insert_mechanisms([[['soma', '0.5', channel, 'gslowbar']]])
    cell.update_attr(['soma', '0.5', channel, 'gslowbar'], 1.0)
    sec_channel = getattr(cell.soma(.5), channel)

    # compute response to voltage steps
    i_steps = []
    i_steps_control, t = voltage_steps(cell.soma, amps, durs, v_steps, stepamp, pos, dt)
    setattr(sec_channel, 'gfastbar', 0.0)
    setattr(sec_channel, 'gslowbar', 0.0)
    i_steps_blockade, t = voltage_steps(cell.soma, amps, durs, v_steps, stepamp, pos, dt)

    for i, v_step in enumerate(v_steps):
        i_steps.append(i_steps_control[i] - i_steps_blockade[i])

    #plot_i_steps(i_steps, v_steps, t)

    # compare to experimental data
    all_traces = pd.read_csv(os.path.join('.', 'plots', 'digitized_vsteps2', 'traces.csv'), index_col=0)
    all_traces /= np.max(np.max(np.abs(all_traces)))

    scale_fac = 1.0 / np.max(np.abs(np.matrix(i_steps)[:, 1:]))
    pl.figure()
    for i, column in enumerate(all_traces.columns):
        pl.plot(all_traces.index, all_traces[column], 'k', label=column)
        pl.plot(t[:-1], i_steps[i][1:] * scale_fac, 'r')
    pl.ylabel('Current (pA)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    #pl.legend(fontsize=16)
    pl.show()
from __future__ import division
from nrn_wrapper import Cell
import numpy as np
from test_channels.test_ionchannel import current_subtraction
import os
import pandas as pd
import matplotlib.pyplot as pl

if __name__ == "__main__":

    # channel to investigate
    channel = "nap_magistretti"
    model_dir = '../../../model/cells/dapmodel_nocurrents.json'
    mechanism_dir = './mod/'

    # parameters
    celsius = 24
    amps = [0, 0, 0]
    durs = [0, 480, 0]
    v_steps = np.arange(-60, -34, 5)
    stepamp = 2
    pos = 0.5
    dt = 1

    # create cell
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    cell.insert_mechanisms([[['soma', '0.5', channel, 'gbar']]])
    cell.update_attr(['soma', '0.5', channel, 'gbar'], 1.0)
    sec_channel = getattr(cell.soma(.5), channel)

    # compute response to voltage steps
    i_steps, t = current_subtraction(cell.soma, sec_channel, celsius, amps, durs, v_steps, stepamp, pos, dt)

    # compare to experimental data
    all_traces = pd.read_csv(os.path.join('.', 'plots', 'digitized_vsteps', 'traces.csv'), index_col=0)
    all_traces /= np.max(np.max(np.abs(all_traces)))

    scale_fac = 1.0 / np.max(np.abs(np.matrix(i_steps)[:, 1:]))
    pl.figure()
    for i, column in enumerate(all_traces.columns):
        pl.plot(all_traces.index, all_traces[column].values, 'k', label='Experiments' if i == 0 else None)
        pl.plot(t[:-1], i_steps[i][1:] * scale_fac, 'r', label='Model Magistretti' if i == 0 else None)
    pl.ylabel('Current (normalized)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.legend(fontsize=16, loc='lower right')
    pl.show()
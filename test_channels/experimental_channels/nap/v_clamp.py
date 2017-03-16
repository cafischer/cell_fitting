from __future__ import division
from nrn_wrapper import *
from test_channels.test_ionchannel import *
import os
import pandas as pd


if __name__ == "__main__":

    # channel to investigate
    channel = "nap"
    model_dir = '../../../model/cells/dapmodel_nocurrents.json'
    mechanism_dir = './mod/'

    # parameters
    celsius = 24
    amps = [-80, 0, -80]
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
    #for i in range(len(i_steps)):
    #    i_steps[i] /= 20
    #plot_i_steps(i_steps, v_steps, t)

    # compare to experimental data
    all_traces = pd.read_csv(os.path.join('.', 'plots', 'digitized_vsteps', 'traces.csv'), index_col=0)
    all_traces /= np.max(np.max(np.abs(all_traces)))

    scale_fac = np.max(np.max(np.abs(all_traces.values))) / np.max(np.abs(i_steps))
    pl.figure()
    for i, column in enumerate(all_traces.columns):
        pl.plot(all_traces.index, all_traces[column], 'k', label=column)
        pl.plot(t, i_steps[i] * scale_fac, 'r')
    pl.ylabel('Current (pA)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.legend(fontsize=16)
    pl.show()
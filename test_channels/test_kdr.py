from __future__ import division
from nrn_wrapper import *
from test_ionchannel import *

__author__ = 'caro'

if __name__ == "__main__":

    # channel to investigate
    channel = "kdr"
    model_dir = '../model/cells/dapmodel_nocurrents.json'
    mechanism_dir = '../model/channels/stellate/'

    # parameters
    celsius = 22
    amps = [-150, -50, -150]
    durs = [150, 150, 150]
    v_steps = np.arange(-50, 100, 10)
    stepamp = 2
    pos = 0.5
    dt = 0.025

    # create cell
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    cell.insert_mechanisms([[['soma', '0.5', channel, 'gbar']]])
    cell.update_attr(['soma', '0.5', channel, 'gbar'], 1.0)
    sec_channel = cell.soma(.5).kdr

    # compute response to voltage steps
    i_steps, t = current_subtraction(cell.soma, sec_channel, celsius, amps, durs, v_steps, stepamp, pos, dt)
    plot_i_steps(i_steps, v_steps, t)
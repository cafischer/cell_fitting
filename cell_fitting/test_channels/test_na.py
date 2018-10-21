from __future__ import division
import numpy as np
from nrn_wrapper import Cell
from test_ionchannel import current_subtraction, plot_i_steps

__author__ = 'caro'

if __name__ == "__main__":

    # channel to investigate
    channel = "nap"
    model_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/2/cell.json'
    mechanism_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/model/channels/vavoulis'

    # create cell
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    #cell.update_attr(['soma', '0.5', channel, 'gbar'], 1.0)
    # cell.update_attr(['soma', '0.5', 'nat', 'gbar'], 0.0)
    cell.update_attr(['soma', '0.5', 'kdr', 'gbar'], 0.0)
    cell.update_attr(['soma', '0.5', 'hcn_slow', 'gbar'], 0.0)
    cell.update_attr(['soma', '0.5', 'pas', 'g'], 0.0)
    sec_channel = getattr(cell.soma(.5), channel)

    # parameters
    celsius = 22
    amps = [-80, 0, -80]
    durs = [20, 20, 100]
    v_steps = np.arange(-70, -20, 10)
    stepamp = 3
    pos = 0.5
    dt = 0.01

    # compute response to voltage steps
    i_steps, t = current_subtraction(cell.soma, sec_channel, celsius, amps, durs, v_steps, stepamp, pos, dt)
    plot_i_steps(i_steps, v_steps, t)

    # parameters
    celsius = 22
    amps = [-90, -80, -80]
    durs = [20, 100, 0]
    v_steps = np.arange(-80, 20, 20)
    stepamp = 2
    pos = 0.5
    dt = 0.01

    # compute response to voltage steps
    i_steps, t = current_subtraction(cell.soma, sec_channel, celsius, amps, durs, v_steps, stepamp, pos, dt)
    plot_i_steps(i_steps, v_steps, t)
from __future__ import division
from nrn_wrapper import *
from test_channels.test_ionchannel import *


if __name__ == "__main__":

    # channel to investigate
    channel = "nap"
    model_dir = '../../../model/cells/dapmodel_nocurrents.json'
    mechanism_dir = './mod/'

    # parameters
    celsius = 24
    amps = [-80, 0, -80]
    durs = [50, 500, 50]
    v_steps = np.linspace(-65, -35, 5)
    stepamp = 2
    pos = 0.5
    dt = 0.025

    # create cell
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    cell.insert_mechanisms([[['soma', '0.5', channel, 'gbar']]])
    cell.update_attr(['soma', '0.5', channel, 'gbar'], 1.0)
    sec_channel = getattr(cell.soma(.5), channel)

    # compute response to voltage steps
    i_steps, t = current_subtraction(cell.soma, sec_channel, celsius, amps, durs, v_steps, stepamp, pos, dt)
    plot_i_steps(i_steps, v_steps, t)
from __future__ import division
from nrn_wrapper import *
from test_channels.test_ionchannel import *


if __name__ == "__main__":

    # channel to investigate
    channel = "hcn"  # "ih" #
    model_dir = '../../../model/cells/dapmodel_nocurrents.json'
    mechanism_dir = './mod/'  # '../../../model/channels/schmidthieber/' #

    # parameters
    celsius = 24
    amps = [0, 1, 0]
    durs = [50, 400, 50]
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
    h.celsius = celsius
    i_steps_control, t = voltage_steps(cell.soma, amps, durs, v_steps, stepamp, pos, dt)
    setattr(sec_channel, 'gfastbar', 0.0)
    setattr(sec_channel, 'gslowbar', 0.0)
    i_steps_blockade, t = voltage_steps(cell.soma, amps, durs, v_steps, stepamp, pos, dt)

    for i, v_step in enumerate(v_steps):
        i_steps.append(i_steps_control[i] - i_steps_blockade[i])

    plot_i_steps(i_steps, v_steps, t)
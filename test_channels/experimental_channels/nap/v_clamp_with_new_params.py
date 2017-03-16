from __future__ import division
from nrn_wrapper import *
from test_channels.test_ionchannel import *


if __name__ == "__main__":

    # channel to investigate
    channel = "nap"
    model_dir = '../../../model/cells/dapmodel_nocurrents.json'
    mechanism_dir = '../../../model/channels/vavoulis/'

    m_tau_min = 0
    m_tau_max = 100.0
    m_tau_delta = 1.0
    h_tau_min = 100
    h_tau_max = 20345.676765
    h_tau_delta = 0.079

    m0 = 0
    h0 = 1
    m_vh = -44.4
    h_vh = -48.8
    m_vs = 5.2
    h_vs = -10

    # parameters
    celsius = 24
    amps = [-80, 0, -80]
    durs = [0, 500, 50]
    v_steps = np.arange(-60, -34, 5)
    stepamp = 2
    pos = 0.5
    dt = 0.025

    # create cell
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    cell.insert_mechanisms([[['soma', '0.5', channel, 'gbar']]])
    cell.update_attr(['soma', '0.5', channel, 'gbar'], 1.0)

    cell.update_attr(['soma', '0.5', channel, 'm_tau_min'], m_tau_min)
    cell.update_attr(['soma', '0.5', channel, 'm_tau_max'], m_tau_max)
    cell.update_attr(['soma', '0.5', channel, 'm_tau_delta'], m_tau_delta)
    cell.update_attr(['soma', '0.5', channel, 'h_tau_min'], h_tau_min)
    cell.update_attr(['soma', '0.5', channel, 'h_tau_max'], h_tau_max)
    cell.update_attr(['soma', '0.5', channel, 'h_tau_delta'], h_tau_delta)

    cell.update_attr(['soma', '0.5', channel, 'm0'], m0)
    cell.update_attr(['soma', '0.5', channel, 'h0'], h0)
    cell.update_attr(['soma', '0.5', channel, 'm_vh'], m_vh)
    cell.update_attr(['soma', '0.5', channel, 'm_vs'], m_vs)
    cell.update_attr(['soma', '0.5', channel, 'h_vh'], h_vh)
    cell.update_attr(['soma', '0.5', channel, 'h_vs'], h_vs)

    sec_channel = getattr(cell.soma(.5), channel)

    # compute response to voltage steps
    i_steps, t = current_subtraction(cell.soma, sec_channel, celsius, amps, durs, v_steps, stepamp, pos, dt)
    plot_i_steps(i_steps, v_steps, t)
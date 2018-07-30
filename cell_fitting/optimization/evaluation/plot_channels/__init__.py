from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from nrn_wrapper import Cell
from cell_fitting.test_channels.channel_characteristics import boltzmann_fun
pl.style.use('paper')


if __name__ == '__main__':

    data_dir = '../../../data/cell_csv_data/2015_08_26b/rampIV/3.1(nA).csv'
    save_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/5'
    model_dir = os.path.join(save_dir, 'cell.json')
    mechanism_dir = '../../../model/channels/vavoulis'

    # load model
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    channel_name = 'nat'

    # get channel parameters
    if channel_name == 'nat' or channel_name == 'nap':
        channel = cell.get_attr(['soma', '0.5', channel_name])
        m_vh = channel.m_vh
        m_vs = channel.m_vs
        h_vh = channel.h_vh
        h_vs = channel.h_vs
        v = np.arange(-100, 20, 0.01)
        act_curve = boltzmann_fun(-v, -m_vh, m_vs)
        inact_curve = boltzmann_fun(-v, -h_vh, h_vs)

        pl.figure()
        pl.plot(v, act_curve, 'r', label='activation')
        pl.plot(v, inact_curve, 'b', label='inactivation')
        pl.ylabel('Open probability')
        pl.xlabel('Membrane potential (mV)')
        pl.legend()
        pl.tight_layout()
        pl.show()

    elif channel_name == 'kdr':
        channel = cell.get_attr(['soma', '0.5', channel_name])
        n_vh = channel.n_vh
        n_vs = channel.n_vs
        v = np.arange(-100, 20, 0.01)
        act_curve = boltzmann_fun(-v, -n_vh, n_vs)

        pl.figure()
        pl.plot(v, act_curve, 'r', label='activation')
        pl.ylabel('Open probability')
        pl.xlabel('Membrane potential (mV)')
        pl.legend()
        pl.tight_layout()
        pl.show()

    elif channel_name == 'hcn_slow':
        channel = cell.get_attr(['soma', '0.5', channel_name])
        n_vh = channel.n_vh
        n_vs = channel.n_vs
        v = np.arange(-100, 20, 0.01)
        inact_curve = boltzmann_fun(-v, -n_vh, n_vs)

        pl.figure()
        pl.plot(v, inact_curve, 'b', label='inactivation')
        pl.ylabel('Open probability')
        pl.xlabel('Membrane potential (mV)')
        pl.legend()
        pl.tight_layout()
        pl.show()
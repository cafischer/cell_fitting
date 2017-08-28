from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from nrn_wrapper import Cell
from optimization.simulate import currents_given_v

__author__ = 'caro'

if __name__ == "__main__":

    # parameter
    data_dir = '../data/2015_08_06d/correct_vrest_-16mV/shortened/PP(0)(3)/0(nA).csv'
    channel = 'nap_act'
    ion = 'na'
    model_dir = '../model/cells/dapmodel_simpel.json'
    mechanism_dir = '../model/channels/vavoulis/'
    celsius = 22

    # create cell
    cell = Cell.from_modeldir(model_dir, mechanism_dir)
    cell.insert_mechanisms([[['soma', '0.5', channel, 'gbar']]])
    cell.update_attr(['soma', '0.5', channel, 'gbar'], 1.0)
    minf = cell.soma.record_from(channel, 'minf')

    # load data
    data = pd.read_csv(data_dir)

    # compute response to voltage
    i_channel = currents_given_v(data.v.values, data.t.values, cell.soma, [channel], [ion], celsius, True)

    pl.figure()
    pl.plot(data.t, np.array(minf))
    pl.show()
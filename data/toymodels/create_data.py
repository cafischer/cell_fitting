import pandas as pd
import matplotlib.pyplot as pl

from nrn_wrapper import Cell, complete_mechanismdir
from optimization.simulate import iclamp, extract_simulation_params

__author__ = 'caro'

data_real_dir = '../2015_08_11d/ramp/ramp.csv'
data_new_dir = './ramp.csv'
model_dir = '../../model/cells/dapmodel0.json'
mechanism_dir = '../../model/channels/schmidthieber'

data_real = pd.read_csv(data_real_dir)

# create cell and run simulation
cell = Cell.from_modeldir(model_dir, complete_mechanismdir(mechanism_dir))
simulation_params = extract_simulation_params(data_real)
v, t = iclamp(cell, **simulation_params)
i = data_real.i.values
sec = data_real.sec.values

data = pd.DataFrame({'v': v, 't': t, 'i': i, 'sec': sec})
data.to_csv(data_new_dir, index=None)

pl.figure()
pl.plot(t, v)
pl.show()

pl.figure()
pl.plot(t, i)
pl.show()

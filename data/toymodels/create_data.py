import pandas as pd
import matplotlib.pyplot as pl
from nrn_wrapper import Cell, iclamp
from optimization.simulate import extract_simulation_params

__author__ = 'caro'

data_real_dir = '../toymodels/hhCell/ramp2.csv'
data_new_dir = '../toymodels/hhCell/ramp2.csv'
model_dir = '../../model/cells/competition_model.json'
mechanism_dir = '../../model/channels/hodgkinhuxley'

data_real = pd.read_csv(data_real_dir)

# create cell and run simulation
cell = Cell.from_modeldir(model_dir, mechanism_dir)
simulation_params = extract_simulation_params(data_real)
simulation_params['celsius'] = 6.3
v, t = iclamp(cell, **simulation_params)
i = data_real.i.values

data = pd.DataFrame({'v': v, 't': t, 'i': i})
data.to_csv(data_new_dir, index=None)

pl.figure()
pl.plot(t, data.v, 'k')
pl.plot(t, v, 'r')
pl.show()

pl.figure()
pl.plot(t, i)
pl.show()

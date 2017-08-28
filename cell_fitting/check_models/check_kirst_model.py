import matplotlib.pyplot as pl
from nrn_wrapper import Cell
from optimization.simulate import extract_simulation_params, iclamp_handling_onset
import pandas as pd


# create Cell
cell = Cell.from_modeldir('../model/cells/kirst_model.json', '../model/channels/kirst')

# load data
data = pd.read_csv('../data/2015_08_06d/correct_vrest_-16mV/IV/1.0(nA).csv')
sim_params = extract_simulation_params(data)

# simulate
sim_params['i_inj'] *= 1900
v, t, i = iclamp_handling_onset(cell, **sim_params)

# plot
pl.figure()
pl.plot(data.t, data.v, 'k')
pl.plot(t, v, 'r')
#pl.show()

# load data
data = pd.read_csv('../data/2015_08_06d/correct_vrest_-16mV/shortened/PP(0)(3)/0(nA).csv')
sim_params = extract_simulation_params(data)

# simulate
sim_params['i_inj'] *= 1900
v, t, i = iclamp_handling_onset(cell, **sim_params)

# plot
pl.figure()
pl.plot(data.t, data.v, 'k')
pl.plot(t, v, 'r')
pl.show()
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

from optimization.bio_inspired.problems import CellFitProblem
from optimization.simulate import run_simulation

__author__ = 'caro'

# TODO
def get_ionlist(channel_list):
    ion_list = []
    for channel in channel_list:
        if 'na' in channel:
            ion_list.append('na')
        elif 'k' in channel:
            ion_list.append('k')
        elif 'ca' in channel:
            ion_list.append('ca')
        else:
            ion_list.append('')
    return ion_list

candidate = [0.25299342]

variables = [
            [0, 2.5, [['soma', '0.5', 'na8st', 'gbar']]]
            ]

params = {
          'maximize': False,
          'normalize': False,
          'model_dir': '../../../model/cells/toymodel1.json',
          'mechanism_dir': '../../../model/channels/schmidthieber',
          'variables': variables,
          'data_dir': '../../../data/toymodels/toymodel1/ramp.csv',
          'get_var_to_fit': 'get_v',
          'fitnessweights': [1.0],
          'errfun': 'rms',
          'insert_mechanisms': True
         }

# create problem
problem = CellFitProblem(**params)

#TODO
#from data import change_dt
#from optimization.bio_inspired.problems import extract_simulation_params
#dt = problem.simulation_params['dt']
#dt_new = dt / 16
#data = pd.read_csv(params['data_dir'])
#data_new = change_dt(dt_new, data)
#problem.data_to_fit[0] = data_new.v.values
#problem.simulation_params = extract_simulation_params(data_new)

# create cell
cell = problem.get_cell(candidate)

# record currents
channel_list = list(set([problem.path_variables[i][0][2] for i in range(len(problem.path_variables))]))
ion_list = get_ionlist(channel_list)
currents = np.zeros(len(channel_list), dtype=object)
for i in range(len(channel_list)):
    currents[i] = cell.soma.record_from(channel_list[i], 'i'+ion_list[i], pos=.5)

# run simulation
#problem.simulation_params['v_init'] = -70
v_model, t = run_simulation(cell, **problem.simulation_params)

# plot
pl.figure()
pl.plot(t, problem.data_to_fit[0], 'k', label='data')
pl.plot(t, v_model, 'r', label='model')
pl.legend()
pl.show()

for i, current in enumerate(currents):
    pl.figure()
    pl.plot(t, current, 'k', label=channel_list[i])
    pl.legend()
    pl.show()

from optimization.bio_inspired.errfuns import rms
print rms(problem.data_to_fit[0], v_model)

# save data
#data = pd.DataFrame({'v': v_model, 't': t, 'i': problem.simulation_params['i_amp'], 'sec': np.nan})
#data.sec.ix[0] = 'soma'
#data.to_csv('../../../data/toymodels/toymodel1/ramp_dt.csv', index=False)
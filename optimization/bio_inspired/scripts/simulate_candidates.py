import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

from optimization.problems import CellFitProblem, get_ionlist
from optimization.simulate import run_simulation

__author__ = 'caro'


candidate = [10]

variables = [
            #[0, 2.5, [['soma', '0.5', 'hh', 'gnabar']]]
            [0, 2.5, [['soma', 'diam']]]
            ]

params = {
          'name': 'CellFitProblem',
          'maximize': False,
          'normalize': False,
          'model_dir': '../../../model/cells/hhCell.json',
          'mechanism_dir': '../../../model/channels/hodgkinhuxley',
          'variables': variables,
          'data_dir': '../../../data/2015_08_11d/ramp/dap.csv',
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
problem.update_cell(candidate)

# record currents
#channel_list = list(set([problem.path_variables[i][0][2] for i in range(len(problem.path_variables))]))
#ion_list = ['na']  #get_ionlist(channel_list)
#currents = np.zeros(len(channel_list), dtype=object)
#for i in range(len(channel_list)):
#    currents[i] = cell.soma.record_from(channel_list[i], 'i'+ion_list[i], pos=.5)

# run simulation
problem.simulation_params['v_init'] = -65
problem.simulation_params['i_amp'] *= 3
problem.simulation_params['celsius'] = 6.3
problem.simulation_params['tstop'] = 40
v_model, t = run_simulation(problem.cell, **problem.simulation_params)

# plot
pl.figure()
#pl.plot(t, problem.data_to_fit[0], 'k', label='data')
pl.plot(t, v_model, 'r', label='model')
pl.legend()
pl.show()

#for i, current in enumerate(currents):
#    pl.figure()
#    pl.plot(t, current, 'k', label=channel_list[i])
#    pl.legend()
#    pl.show()

#from optimization.errfuns import rms
#print rms(problem.data_to_fit[0], v_model)

# save data
import os
#path = '../../../data/toymodels/hhCell/'
#if not os.path.exists(path):
#    os.makedirs(path)
#data = pd.DataFrame({'v': v_model, 't': t, 'i': problem.simulation_params['i_amp'][:len(t)]})
#data.to_csv(path+'ramp.csv', index=False)
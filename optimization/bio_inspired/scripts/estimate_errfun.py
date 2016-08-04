import copy
import os
import json

import numpy as np

from optimization.problems import CellFitProblem
from optimization.simulate import run_simulation
from optimization.errfuns import rms

__author__ = 'caro'

# parameter
save_dir = '../../../results/visualize_errfun/test_algorithms/kinetic_param/toymodel3/'
theta_init = [0, 0]
p1_idx = 0
p2_idx = 1
p1_range = np.arange(0, 2.5, 0.01)
p2_range = np.arange(0, 0.1, 0.0001)

variables = [
            [0, 2.5, [['soma', '0.5', 'na8st', 'gbar']]],
            [0, 2.5, [['soma', '0.5', 'na8st', 'a1_1']]]
            ]

params = {
          'maximize': False,
          'normalize': True,
          'model_dir': '../../../model/cells/toymodel3.json',
          'mechanism_dir': '../../../model/channels/schmidthieber',
          'variables': variables,
          'data_dir': '../../../data/toymodels/toymodel3/ramp.csv',
          'get_var_to_fit': 'get_v',
          'fitnessweights': [1.0],
          'errfun': 'rms',
          'insert_mechanisms': True
         }

# create problem
problem = CellFitProblem(**params)

# compute error
error = np.zeros((len(p1_range), len(p2_range)))
for i, p1 in enumerate(p1_range):
    for j, p2 in enumerate(p2_range):
        theta = copy.copy(theta_init)
        theta[p1_idx] = p1
        theta[p2_idx] = p2

        # run simulation with these parameters
        problem.update_cell(theta)
        v_model, t = run_simulation(problem.cell, **problem.simulation_params)
        error[i, j] = rms(problem.data_to_fit[0], v_model)

# save error
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(save_dir+'/err_'+variables[p1_idx][2][0][2]+variables[p1_idx][2][0][3]
                  +variables[p2_idx][2][0][2]+variables[p2_idx][2][0][3]+'.npy', 'w') as f:
    np.save(f, error)
with open(save_dir+'params.json', 'w') as f:
    json.dump(params, f)
np.savetxt(save_dir+'theta_init.txt', theta_init)
np.savetxt(save_dir+'p1_range.txt', p1_range)
np.savetxt(save_dir+'p2_range.txt', p2_range)
np.savetxt(save_dir+'p1_idx.txt', np.array([p1_idx]))
np.savetxt(save_dir+'p2_idx.txt', np.array([p2_idx]))

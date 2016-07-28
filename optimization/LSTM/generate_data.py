from random import Random
from time import time
import numpy as np
import os
import json
from nrn_wrapper import Cell

from optimization.bio_inspired.problems import CellFitProblem
from optimization.simulate import run_simulation


__author__ = 'caro'

# parameter
n_data = 100
save_dir = './data/toymodel1/na8st/3/'

# specify model
variables = [
            [0, 0.5, [['soma', '0.5', 'na8st', 'gbar']]]
            ]

params = {
          'maximize': False,
          'normalize': False,
          'model_dir': '../../model/cells/toymodel1.json',
          'mechanism_dir': '../../model/channels/schmidthieber',
          'variables': variables,
          'data_dir': '../../data/toymodels/toymodel1/ramp.csv',
          'get_var_to_fit': 'get_v',
          'fitnessweights': [1.0],
          'errfun': 'rms',
          'insert_mechanisms': True
         }

problem = CellFitProblem(**params)
dt = problem.simulation_params['dt']
time_cut = 20
problem.simulation_params['tstop'] = time_cut  # TODO
problem.simulation_params['i_amp'] = problem.simulation_params['i_amp'][:time_cut/dt+1]
subsample = 5

# initialize pseudo random number generator
seed = time()
prng = Random()
prng.seed(seed)

# initialize data
t = np.arange(0, problem.simulation_params['tstop']+dt, dt)
i_inj = problem.simulation_params['i_amp']
data = np.zeros((n_data, len(t)/subsample, 2))  # inputs are the trace of v and i_inj
labels = np.zeros(n_data)

for i in range(n_data):
    # modify parameter
    candidate = problem.generator(prng, None)

    # run simulation
    cell = problem.get_cell(candidate)
    v, t = run_simulation(cell, **problem.simulation_params)
    data[i, :, 0] = v[:-1:subsample]
    data[i, :, 1] = i_inj[:-1:subsample]
    labels[i] = candidate[0]

# store data
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(save_dir+'/problem.json', 'w') as f:
    json.dump(params, f, indent=4)
with open(save_dir+'/cell.json', 'w') as f:
    json.dump(Cell.from_modeldir(params['model_dir']).get_dict(), f, indent=4)
np.savetxt(save_dir+'/seed.txt', np.array([seed]))
with open(save_dir+'data.npy', 'w') as f:
    np.save(f, data)
with open(save_dir+'labels.npy', 'w') as f:
    np.save(f, labels)
from random import Random
from time import time
import numpy as np
import os
import json
from nrn_wrapper import Cell
from optimization.bio_inspired.problems import CellFitProblem
from scipy.optimize import minimize
import functools
from optimization.gradient_descent import numerical_gradient

__author__ = 'caro'


# parameter
#method = 'Nelder-Mead'
#method = 'BFGS'
method = 'Newton-CG'
save_dir = '../../../results/gradient_descent/test_algorithms/increase_params/2param/'+method+'/'
n_trials = 20

variables = [
            [0, 2.5, [['soma', '0.5', 'na8st', 'gbar']]],
            [0, 2.5, [['soma', '0.5', 'kdr', 'gbar']]]
            ]

params = {
          'maximize': False,
          'normalize': False,
          'model_dir': '../../../model/cells/toymodel3.json',
          'mechanism_dir': '../../../model/channels/schmidthieber',
          'variables': variables,
          'data_dir': '../../../data/toymodels/toymodel3/ramp.csv',
          'get_var_to_fit': 'get_v',
          'fitnessweights': [1.0],
          'errfun': 'rms',
          'insert_mechanisms': True
         }

problem = CellFitProblem(**params)

# save all information
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(save_dir+'/problem.json', 'w') as f:
    json.dump(params, f, indent=4)
with open(save_dir+'/cell.json', 'w') as f:
    json.dump(Cell.from_modeldir(params['model_dir']).get_dict(), f, indent=4)

for trial in range(0, n_trials):
    if os.path.isfile(save_dir+'/seed'+str(trial)+'.txt'):
        seed = float(np.loadtxt(save_dir+'seed_'+str(trial)+'.txt'))
    else:
        seed = time()
        np.savetxt(save_dir+'/seed_'+str(trial)+'.txt', np.array([seed]))
    prng = Random()
    prng.seed(seed)
    candidate = problem.generator(prng, None)

    f = functools.partial(problem.evaluate, args=None)
    fprime = functools.partial(numerical_gradient, f=f, h=1e-10, method='central')
    xopt = minimize(f, candidate, method=method, jac=fprime)
    print 'start value: ' + str(candidate)
    print 'end value: ' + str(xopt.x)
    np.savetxt(save_dir+'/best_candidate_'+str(trial)+'.txt', xopt.x)
    np.savetxt(save_dir+'/error_'+str(trial)+'.txt', np.array([xopt.fun]))
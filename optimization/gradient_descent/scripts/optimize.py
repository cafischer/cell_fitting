import functools
import os

import numpy as np
import matplotlib.pyplot as pl

from optimization.simulate import run_simulation
from optimization.gradient_descent import numerical_gradient
from optimization.bio_inspired.problems import CellFitProblem
from optimization.bio_inspired.errfuns import rms

__author__ = 'caro'

# parameter
save_dir = '../../../results/gradient_descent/test'
theta_init = np.array([0.01874379, 52.3, 6.8, 1, 55])
learn_rate = 1e-10
h = learn_rate
num_iterations = 1
args = []
kwargs = {}

variables = [
            [0, 0.2, [['soma', '0.5', 'nap', 'gbar']]],
            [30, 70, [['soma', '0.5', 'nap', 'sh']]],
            [1, 15, [['soma', '0.5', 'nap', 'k']]],
            [0.1, 5, [['soma', '0.5', 'nap', 'scalerate']]],
            [50, 65, [['soma', '0.5', 'nap', 'eNa']]]
            ]
params = {
          'maximize': False,
          'normalize': False,
          'model_dir': '../../../model/cells/dapmodel0.json',
          'mechanism_dir': '../../../model/channels/schmidthieber',
          'variables': variables,
          'data_dir': '../../../data/2015_08_11d/ramp/dap.csv',
          'get_var_to_fit': 'get_v',
          'fitnessweights': [1],
          'errfun': 'rms',
          'insert_mechanisms': False
         }
problem = CellFitProblem(**params)
fun = functools.partial(problem.evaluate, args=None)
fun_gradient = functools.partial(numerical_gradient, f=fun, h=1e-10, method='central')

# gradient descent
#theta_new = adagrad_single(theta_init, fun_gradient, num_iterations, problem.lower_bound, problem.upper_bound,
#                          gamma=0.9, eps=1e-8)
#theta_new = gradientdescent_single(theta_init, learn_rate, fun_gradient, num_iterations, problem.lower_bound,
#                                   problem.upper_bound)

theta_new = [1.874379438669923278e-02, 5.230000000000710259e+01, 6.800000000005379519e+00, 9.999999999975375253e-01, 5.500000000000000000e+01]

# save
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.savetxt(save_dir+'/theta_new.txt', theta_new)

# plot before and after gradient descent
v_model_before, t = run_simulation(problem.get_cell(theta_init), **problem.simulation_params)
v_model_after, t = run_simulation(problem.get_cell(theta_new), **problem.simulation_params)
print rms(problem.data_to_fit[0], v_model_before)
print rms(problem.data_to_fit[0], v_model_after)
pl.figure()
pl.plot(t, problem.data_to_fit[0], 'k', label='data')
pl.plot(t, v_model_before, 'r', label='model before descent')
pl.plot(t, v_model_after, 'b', label='model after descent')
pl.legend()
pl.show()

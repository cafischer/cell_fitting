from new_optimization.fitter.testfitter import *
from new_optimization.optimizer.scipy_optimizer import *
import os
from time import time
import numpy as np

save_dir = '../../results/fitnesslandscapes/follow_max_gradient2/gna_gk/test/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

maximize = False
n_candidates = 1
stop_criterion = ['generation_termination', 100]
seed = 1.11  #time()
generator = 'get_random_numbers_in_bounds'
bounds = {'lower_bounds': [-10], 'upper_bounds': [10]}
fitfuns = [lambda x: x[0]**4-4*x[0]**2+4, lambda x: np.abs(x[0]-np.sqrt(2))]
fitter = TestFitter(fitfuns)

algorithm_name = 'CG'
algorithm_params = {'step': 1e-8}
normalize = False

optimization_params = None
save_dir_results = save_dir + '/'+algorithm_name+'/'
if not os.path.exists(save_dir_results):
    os.makedirs(save_dir_results)

optimization_settings = OptimizationSettings(maximize, n_candidates, stop_criterion, seed, generator, bounds, fitter)
with open(save_dir_results+'/optimization_settings.json', 'w') as f:
    optimization_settings.save(f)

algorithm_settings = AlgorithmSettings(algorithm_name, algorithm_params, optimization_params, normalize, save_dir_results)
with open(save_dir_results+'/algorithm_settings.json', 'w') as f:
    algorithm_settings.save(f)

optimizer = ScipyMaxOptimizer(optimization_settings, algorithm_settings)
optimizer.optimize()

x = np.arange(-10, 10, 0.01)
import matplotlib.pyplot as pl
points = [-1, -1.4799546632686655, -1.4044380673045342, 1.4142135631837136, 1.4142135623730951, 1.4142135623730951]
point_fun = [0, 0, 1, 1, 1, 1]
pl.figure()
pl.plot(x, [fitfuns[0]([xi]) for xi in x], label='$x^4 - 4x^2 + 4$')
pl.plot(x, [fitfuns[1]([xi]) for xi in x], label='$|x|- \sqrt{2}$')
pl.plot(points, [fitfuns[point_fun[i]]([points[i]]) for i in range(len(points))], 'xk', mew=1)
pl.ylim(-0.5, 4.5)
pl.xlim(-5, 9)
pl.legend()
pl.show()

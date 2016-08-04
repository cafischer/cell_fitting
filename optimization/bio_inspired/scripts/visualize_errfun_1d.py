import json

import matplotlib.pyplot as pl
import numpy as np

from optimization.problems import CellFitProblem

__author__ = 'caro'

save_dir = '../../../results/visualize_errfun/test_algorithms/increase_channels/1channel/'

# load data
p1_range = np.loadtxt(save_dir+'p1_range.txt')
print p1_range[1] - p1_range[0]
p1_idx = int(np.loadtxt(save_dir+'p1_idx.txt'))
with open(save_dir+'params.json', 'r') as f:
    params = json.load(f)
problem = CellFitProblem(**params)
error = np.load(save_dir+'err_'+problem.path_variables[p1_idx][0][2]+problem.path_variables[p1_idx][0][3]+'.npy')

# compute minima
#minima = get_minima_2d(error, 0.0001, 5)
#print 'Number of Minima: ' + str(len(minima))
#print minima

# plot errfun for both parameter
pl.figure()
pl.plot(p1_range, error, '-k', markersize=5)
pl.ylabel(params['errfun'])
pl.xlabel(problem.path_variables[p1_idx][0][2]+' '+problem.path_variables[p1_idx][0][3])
pl.xlim(p1_range[0], p1_range[-1])
pl.savefig(save_dir+'err_'+problem.path_variables[p1_idx][0][2]+problem.path_variables[p1_idx][0][3]+'.png')
pl.show()

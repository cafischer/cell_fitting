import json

import matplotlib.pyplot as pl
import numpy as np

from optimization.problem import CellFitProblem

__author__ = 'caro'

import scipy.signal


def get_minima_2d(mat, comp_val, order, mode='clip'):
    def comp(x, y):
        return y-x > comp_val
    min_x = scipy.signal.argrelextrema(mat, comp, axis=0, order=order, mode=mode)
    min_y = scipy.signal.argrelextrema(mat, comp, axis=1, order=order, mode=mode)

    min_x = [tuple(np.array(min_x)[:, j]) for j in range(np.shape(np.array(min_x))[1])]  # transform to tuples with (x, y)
    min_y = [tuple(np.array(min_y)[:, j]) for j in range(np.shape(np.array(min_y))[1])]
    minima = set(min_x).intersection(set(min_y))
    return minima

save_dir = '../../../results/visualize_errfun/test_algorithms/kinetic_param/toymodel3/na8st_gbar_a3_1/'

# load data
p1_range = np.loadtxt(save_dir+'p1_range.txt')
print p1_range[1] - p1_range[0]
p2_range = np.loadtxt(save_dir+'p2_range.txt')
p1_idx = int(np.loadtxt(save_dir+'p1_idx.txt'))
p2_idx = int(np.loadtxt(save_dir+'p2_idx.txt'))
with open(save_dir+'params.json', 'r') as f:
    params = json.load(f)
problem = CellFitProblem(**params)

error = np.load(save_dir+'err_'+problem.path_variables[p1_idx][0][2]+problem.path_variables[p1_idx][0][3]
                +problem.path_variables[p2_idx][0][2]+problem.path_variables[p2_idx][0][3]+'.npy')

# compute minima
#minima = get_minima_2d(error, 0.0001, 5)
#print 'Number of Minima: ' + str(len(minima))
#print minima

# plot errfun for both
print error[p1_range==0.04, p2_range==0.012]
X, Y = np.meshgrid(p1_range, p2_range)
pl.figure()
pl.pcolormesh(X, Y, error.T)
pl.colorbar()
pl.xlabel(problem.path_variables[p1_idx][0][2]+' '+problem.path_variables[p1_idx][0][3])
pl.ylabel(problem.path_variables[p2_idx][0][2]+' '+problem.path_variables[p2_idx][0][3])
pl.xlim(p1_range[0], p1_range[-1])
pl.ylim(p2_range[0], p2_range[-1])
pl.savefig(save_dir+'err_'+problem.path_variables[p1_idx][0][2]+problem.path_variables[p2_idx][0][2]+'.png')
pl.show()

import numpy as np
import os
import matplotlib.pyplot as pl
from optimization.fitnesslandscape import *

save_dir = '../../results/modellandscape/hhCell/gna_gk/'
new_folder = 'fitfuns/APamp+v_trace+penalize_not1AP'
folders = ['fitfuns/APamp', 'fitfuns/v_trace+penalize_not1AP']
fitfuns = ['APamp', 'v_trace+penalize_not1AP']
order = 1
optimum = [0.12, 0.036]

errors = list()
for folder in folders:
    with open(save_dir + '/' + folder + '/error.npy', 'r') as f:
        error = np.load(f)
        error[np.isnan(error)] = 0
        errors.append(error)
    with open(save_dir + '/' + folder + '/has_1AP.npy', 'r') as f:
        has_1AP_mat = np.load(f)
p1_range = np.loadtxt(save_dir + '/p1_range.txt')
p2_range = np.loadtxt(save_dir + '/p2_range.txt')

error = np.array(map(np.add, *errors))

if not os.path.exists(save_dir + '/' + new_folder + '/'):
    os.makedirs(save_dir + '/' + new_folder + '/')
with open(save_dir + '/' + new_folder + '/error.npy', 'w') as f:
    np.save(f, error)
with open(save_dir + '/' + new_folder + '/has_1AP.npy', 'w') as f:
    np.save(f, has_1AP_mat)

minima_xy = get_local_minima_2d(error, order)

P1, P2 = np.meshgrid(p1_range, p2_range)
fig, ax = pl.subplots()
im = ax.pcolormesh(P1, P2, np.ma.masked_invalid(error).T)
for minimum in minima_xy:
    if has_1AP_mat[minimum[0], minimum[1]]:
        ax.plot(p1_range[minimum[0]], p2_range[minimum[1]], 'ow')
    else:
        ax.plot(p1_range[minimum[0]], p2_range[minimum[1]], 'ok')
ax.plot(optimum[0], optimum[1], 'x', color='k', mew=2, ms=8)
pl.xlabel('gmax na')
pl.ylabel('gmax k')
pl.title(str(fitfuns))
fig.colorbar(im)
pl.savefig(save_dir + '/' + new_folder + '/fitness_landscape_' + str(order) + '.png')
pl.show()
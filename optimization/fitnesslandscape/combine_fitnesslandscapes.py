import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from optimization.fitnesslandscape import *

save_dir = '../../results/fitnesslandscapes/modellandscape/gna_gk_highresolution/'
new_folder = 'fitfuns/APamp+v_trace'
folders = ['fitfuns/APamp', 'fitfuns/v_trace']
fitfuns = ['APamp', 'v_trace']
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

error = np.array(map(np.maximum, *errors))

if not os.path.exists(save_dir + '/' + new_folder + '/'):
    os.makedirs(save_dir + '/' + new_folder + '/')
with open(save_dir + '/' + new_folder + '/error.npy', 'w') as f:
    np.save(f, error)
with open(save_dir + '/' + new_folder + '/has_1AP.npy', 'w') as f:
    np.save(f, has_1AP_mat)

minima2d = get_local_minima_2d(error)
with open(save_dir + '/' + new_folder + '/minima2d.npy', 'w') as f:
    np.save(f, np.array(minima2d))

P1, P2 = np.meshgrid(p1_range, p2_range)
fig, ax = pl.subplots()
im = ax.pcolormesh(P1, P2, np.ma.masked_invalid(error).T)
for minimum in minima2d:
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
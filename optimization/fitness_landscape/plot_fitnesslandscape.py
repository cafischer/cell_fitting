import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as pl
from optimization.fitness_landscape import *

save_dir = '../../../results/modellandscape/hhCell/zoom_gna_gk/'
new_name = 'rms_inside'
optimum = [0.12, 0.036]

with open(save_dir + '/' + new_name + '/error.npy', 'r') as f:
    error = np.load(f)
with open(save_dir + '/' + new_name + '/has_AP.npy', 'r') as f:
    has_AP_mat = np.load(f)
p1_range = np.loadtxt(save_dir + '/p1_range.txt')
p2_range = np.loadtxt(save_dir + '/p2_range.txt')

order = 1

minima_x = list()
minima_y = list()
minima_xy = list()

for i in range(np.shape(error)[0]):
    #minima_x.extend([[i, y] for y in get_local_minima(error[i, :], order=order)])
    minima_x.extend([[i, y] for y in get_true_local_minima(error[i, :], order=order)])

for i in range(np.shape(error)[1]):
    #minima_y.extend([[x, i] for x in get_local_minima(error[:, i], order=order)])
    minima_y.extend([[x, i] for x in get_true_local_minima(error[:, i], order=order)])

# problem with several dimensions: if several points at the trough minima can be overlooked
for minimum in minima_x:
    if minimum in minima_y:
        minima_xy.append(minimum)

print minima_xy
print len(minima_xy)

P1, P2 = np.meshgrid(p1_range, p2_range)
fig, ax = pl.subplots()
im = ax.pcolormesh(P1, P2, ma.masked_invalid(error).T)
for minimum in minima_xy:
    if has_AP_mat[minimum[0], minimum[1]]:
        ax.plot(p1_range[minimum[0]], p2_range[minimum[1]], 'ow')
    else:
        ax.plot(p1_range[minimum[0]], p2_range[minimum[1]], 'ok')
ax.plot(optimum[0], optimum[1], 'xk')
pl.xlabel('gmax na')
pl.ylabel('gmax k')
pl.title(new_name)
fig.colorbar(im)
pl.savefig(save_dir+'/'+new_name+'/fitness_landscape_'+str(order)+'.png')
#pl.show()

fig, ax = pl.subplots()
im = ax.pcolormesh(P1, P2, ma.masked_invalid(has_AP_mat).T)
ax.plot(optimum[0], optimum[1], 'xk')
pl.xlabel('gmax na')
pl.ylabel('gmax k')
pl.title('Models with 1 AP')
fig.colorbar(im)
pl.savefig(save_dir+'/'+new_name+'/1AP.png')
pl.show()
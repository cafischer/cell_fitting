import numpy as np
import matplotlib.pyplot as pl
from optimization.errfuns import rms
from optimization.fitfuns import *
import pandas as pd
import json
import numpy.ma as ma
from optimization.error_landscape_measures import *

save_dir = '../../../results/modellandscape/hhCell/gna_gk/'
new_name = 'test_rms'

with open(save_dir+'/modellandscape.npy', 'r') as f:
    modellandscape = np.load(f)
p1_range = np.loadtxt(save_dir + '/p1_range.txt')
p2_range = np.loadtxt(save_dir + '/p1_range.txt')
with open(save_dir + '/dirs.json', 'r') as f:
    dirs = json.load(f)
data = pd.read_csv(dirs['data_dir'])

optimum = [0.12, 0.036]

error = np.zeros((np.shape(modellandscape)[0], np.shape(modellandscape)[1]))

threshold = -30
window_before = 5
window_after = 20

args_data = {'shift': 0, 'window_before': window_before, 'window_after': window_after, 'threshold': threshold}
AP_time_data = get_APtime(np.array(data.v), np.array(data.t), np.array(data.i), args_data)
args_data['APtime'] = AP_time_data[0]
AP_amp_data = get_APamp(np.array(data.v), np.array(data.t), np.array(data.i), args_data)
AP_width_data = get_APwidth(np.array(data.v), np.array(data.t), np.array(data.i), args_data)
AP_window_data = shifted_AP(np.array(data.v), np.array(data.t), np.array(data.i), args_data)
APmax_data = np.max(np.array(data.v))

exp_data = np.array(data.v)  #AP_window_data

args = {'shift': 4, 'window_before': window_before, 'window_after': window_after, 'APtime': AP_time_data[0],
          'threshold': threshold}

for i in range(np.shape(modellandscape)[0]):
    for j in range(np.shape(modellandscape)[1]):
        model_data = get_v(modellandscape[i, j], data.t, data.i, args)

        if None in model_data:
            error[i, j] = None
        else:
            error[i, j] = rms(model_data, exp_data)


order = 1

minima_x = list()
minima_y = list()
minima_xy = list()

for i in range(np.shape(error)[0]):
    minima_x.extend([[i, y] for y in get_number_local_minima(error[i, :], order=order)])

for i in range(np.shape(error)[1]):
    minima_y.extend([[x, i] for x in get_number_local_minima(error[:, i], order=order)])

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
    ax.plot(p1_range[minimum[0]], p2_range[minimum[1]], 'ok')
ax.plot(optimum[0], optimum[1], 'xk')
pl.xlabel('gmax na')
pl.ylabel('gmax k')
pl.title(new_name)
fig.colorbar(im)
pl.savefig(save_dir+'/'+new_name+'/fitness_landscape_'+str(order)+'.png')
pl.show()
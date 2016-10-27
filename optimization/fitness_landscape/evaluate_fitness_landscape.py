import numpy as np
import matplotlib.pyplot as pl
from optimization.errfuns import rms
from optimization.fitfuns import *
import pandas as pd
import json
import numpy.ma as ma
from optimization.fitness_landscape import *

save_dir = '../../../results/modellandscape/hhCell/gna_gk/'
save_dir_minima = '../../../results/fitness_landscape/find_local_minima/gna_gk/APamp/trust-ncg/'
new_name = 'APamp_inside'

with open(save_dir+'/modellandscape.npy', 'r') as f:
    modellandscape = np.load(f)
p1_range = np.loadtxt(save_dir + '/p1_range.txt')
p2_range = np.loadtxt(save_dir + '/p2_range.txt')
with open(save_dir + '/dirs.json', 'r') as f:
    dirs = json.load(f)
data = pd.read_csv(dirs['data_dir'])

optimum = [0.12, 0.036]

error = np.zeros((np.shape(modellandscape)[0], np.shape(modellandscape)[1]))
has_AP_mat = np.zeros((np.shape(modellandscape)[0], np.shape(modellandscape)[1]))

threshold = -30
window_before = 5
window_after = 20
dt = data.t.values[1] - data.t.values[0]
dvdt = np.concatenate((np.array([(data.v[1] - data.v[0]) / dt]), np.diff(data.v) / dt))

args_data = {'shift': 0, 'window_before': window_before, 'window_after': window_after, 'threshold': threshold,
             'v_min': np.min(data.v), 'v_max': np.max(data.v), 'dvdt_min': np.min(dvdt), 'dvdt_max': np.max(dvdt),
             'bins_v': 100, 'bins_dvdt': 100}
AP_time_data = get_APtime(np.array(data.v), np.array(data.t), np.array(data.i), args_data)
args_data['APtime'] = AP_time_data[0]
AP_amp_data = get_APamp(np.array(data.v), np.array(data.t), np.array(data.i), args_data)
AP_width_data = get_APwidth(np.array(data.v), np.array(data.t), np.array(data.i), args_data)
AP_window_data = shifted_AP(np.array(data.v), np.array(data.t), np.array(data.i), args_data)
APmax_data = np.max(np.array(data.v))
hist_data = phase_hist(np.array(data.v), np.array(data.t), np.array(data.i), args_data)

args = {'shift': 4, 'window_before': window_before, 'window_after': window_after, 'APtime': AP_time_data[0],
        'threshold': threshold,
        'v_min': np.min(data.v), 'v_max': np.max(data.v), 'dvdt_min': np.min(dvdt), 'dvdt_max': np.max(dvdt),
        'bins_v': 100, 'bins_dvdt': 100}

for i in range(np.shape(modellandscape)[0]):
    for j in range(np.shape(modellandscape)[1]):
        has_AP = has_1AP(modellandscape[i, j], data.t, data.i, args)
        has_AP_mat[i, j] = has_AP

        if has_AP:
            if 'rms' in new_name:
                model_data = get_v(modellandscape[i, j], data.t, data.i, args)
                exp_data = get_v(np.array(data.v), data.t, data.i, args)
            elif 'APshift' in new_name:
                model_data = shift_AP_max_APdata(modellandscape[i, j], data.t, data.i, args)
                exp_data = AP_window_data
            elif 'APamp' in new_name:
                model_data = get_APamp(modellandscape[i, j], data.t, data.i, args)
                exp_data = AP_amp_data
            elif 'APwidth' in new_name:
                model_data = get_APwidth(modellandscape[i, j], data.t, data.i, args)
                exp_data = AP_width_data
            elif 'APtime' in new_name:
                model_data = get_APtime(modellandscape[i, j], data.t, data.i, args)
                exp_data = AP_time_data
            elif 'phasehist' in new_name:
                model_data = phase_hist(modellandscape[i, j], data.t, data.i, args)
                exp_data = hist_data
            else:
                raise ValueError('Unknown error function!')
        else:
            model_data = get_v(modellandscape[i, j], data.t, data.i, args)
            exp_data = [np.array(data.v)]

        if None in model_data:
            error[i, j] = None
        else:
            error[i, j] = rms(model_data[0], exp_data[0])


with open(save_dir + '/' + new_name + '/error.npy', 'w') as f:
    np.save(f, error)
with open(save_dir + '/' + new_name + '/has_AP.npy', 'w') as f:
    np.save(f, has_AP_mat)

order = 1

minima_x = list()
minima_y = list()
minima_xy = list()

for i in range(np.shape(error)[0]):
    minima_x.extend([[i, y] for y in get_local_minima(error[i, :], order=order)])

for i in range(np.shape(error)[1]):
    minima_y.extend([[x, i] for x in get_local_minima(error[:, i], order=order)])

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
#pl.show()

with open(save_dir_minima+'/minima_descent.npy', 'r') as f:
    minima_descent = np.load(f)

P1, P2 = np.meshgrid(p1_range, p2_range)
fig, ax = pl.subplots()
im = ax.pcolormesh(P1, P2, ma.masked_invalid(error).T)
for minimum in minima_xy:
    if has_AP_mat[minimum[0], minimum[1]]:
        ax.plot(p1_range[minimum[0]], p2_range[minimum[1]], 'ow')
    else:
        ax.plot(p1_range[minimum[0]], p2_range[minimum[1]], 'ok')
for minimum in minima_descent:
    ax.plot(minimum[0], minimum[1], 'o', color='0.5', markersize=5)
ax.plot(optimum[0], optimum[1], 'xk')
pl.xlabel('gmax na')
pl.ylabel('gmax k')
pl.title(new_name)
fig.colorbar(im)
pl.savefig(save_dir+'/'+new_name+'/fitness_landscape_descent_'+str(order)+'.png')
pl.show()

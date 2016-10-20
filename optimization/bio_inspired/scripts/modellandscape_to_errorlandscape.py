import numpy as np
import matplotlib.pyplot as pl
from optimization.errfuns import rms
from optimization.fitfuns import *
import pandas as pd
import json
import numpy.ma as ma

save_dir = '../../../results/modellandscape/hhCell/gna_gk/'
new_name = 'test_APshift_or_maxshift_or_datashift'

with open(save_dir+'/modellandscape.npy', 'r') as f:
    modellandscape = np.load(f)
p1_range = np.loadtxt(save_dir + '/p1_range.txt')
p2_range = np.loadtxt(save_dir + '/p1_range.txt')
with open(save_dir + '/dirs.json', 'r') as f:
    dirs = json.load(f)
data = pd.read_csv(dirs['data_dir'])

optimum = [0.12, 0.036]  # [0.12, 0.036, 0.0003]

error_rms = np.zeros((np.shape(modellandscape)[0], np.shape(modellandscape)[1]))
error_APamp = np.zeros((np.shape(modellandscape)[0], np.shape(modellandscape)[1]))
error_APwidth = np.zeros((np.shape(modellandscape)[0], np.shape(modellandscape)[1]))
error_APtime = np.zeros((np.shape(modellandscape)[0], np.shape(modellandscape)[1]))
error_APshift_4 = np.zeros((np.shape(modellandscape)[0], np.shape(modellandscape)[1]))
error_APshift_2 = np.zeros((np.shape(modellandscape)[0], np.shape(modellandscape)[1]))

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

args_4 = {'shift': 4, 'window_before': window_before, 'window_after': window_after, 'APtime': AP_time_data[0],
          'threshold': threshold}
args_2 = {'shift': 2, 'window_before': window_before, 'window_after': window_after, 'APtime': AP_time_data[0],
          'threshold': threshold}
args_others = {'threshold': threshold}

for i in range(np.shape(modellandscape)[0]):
    for j in range(np.shape(modellandscape)[1]):

        #pl.figure()
        #pl.plot(data.t, modellandscape[i, j])
        #pl.show()

        AP_amp = get_APamp(modellandscape[i, j], data.t, data.i, args_others)
        AP_width = get_APwidth(modellandscape[i, j], data.t, data.i, args_others)
        AP_time = get_APtime(modellandscape[i, j], data.t, data.i, args_others)

        AP_window_4 = shifted_best(modellandscape[i, j], np.array(data.t), np.array(data.i), args_4)  # TODO
        AP_window_2 = shifted_AP(modellandscape[i, j], np.array(data.t), np.array(data.i), args_2)

        error_rms[i, j] = rms(modellandscape[i, j], data.v)
        if None in AP_amp:
            error_APamp[i, j] = None
        else:
            error_APamp[i, j] = rms(AP_amp, AP_amp_data)
        if None in AP_width:
            error_APwidth[i, j] = None
        else:
            error_APwidth[i, j] = rms(AP_width, AP_width_data)
        if None in AP_time:
            error_APtime[i, j] = None
        else:
            error_APtime[i, j] = rms(AP_time, AP_time_data)
        if None in AP_window_4:
            error_APshift_4[i, j] = None #np.abs(np.max(APmax_data - np.max(modellandscape[i, j])))  # TODO None
        else:
            error_APshift_4[i, j] = rms(AP_window_4, AP_window_data)
        if None in AP_window_2:
            error_APshift_2[i, j] = None
        else:
            error_APshift_2[i, j] = rms(AP_window_2, AP_window_data)

P1, P2 = np.meshgrid(p1_range, p2_range)
pl.figure()
pl.pcolormesh(P1, P2, ma.masked_invalid(error_APshift_4).T)
pl.xlabel('gmax na')
pl.ylabel('gmax k')
pl.colorbar()
pl.title('shift window to AP or maximum or AP data')
pl.savefig(save_dir+'/'+new_name+'/fitness_landscape_APshift.png')
pl.show()


P1, P2 = np.meshgrid(p1_range, p2_range)
fig, ax = pl.subplots(3, 2, sharex=True, sharey=True)
im = ax[0, 0].pcolormesh(P1, P2, error_rms.T / np.max(error_rms))
ax[0, 0].plot(optimum[0], optimum[1], 'xk')
ax[0, 0].set_title('RMS(V, V*)')
ax[0, 1].pcolormesh(P1, P2, ma.masked_invalid(error_APamp).T / np.max(ma.masked_invalid(error_APamp)))
ax[0, 1].plot(optimum[0], optimum[1], 'xk')
ax[0, 1].set_title('AP amplitude')
ax[1, 0].pcolormesh(P1, P2, ma.masked_invalid(error_APwidth).T / np.max(ma.masked_invalid(error_APwidth)))
ax[1, 0].plot(optimum[0], optimum[1], 'xk')
ax[1, 0].set_title('AP width')
ax[1, 1].pcolormesh(P1, P2, ma.masked_invalid(error_APtime).T / np.max(ma.masked_invalid(error_APtime)))
ax[1, 1].plot(optimum[0], optimum[1], 'xk')
ax[1, 1].set_title('AP time')
ax[2, 0].pcolormesh(P1, P2, ma.masked_invalid(error_APshift_4).T / np.max(ma.masked_invalid(error_APshift_4)))
ax[2, 0].plot(optimum[0], optimum[1], 'xk')
ax[2, 0].set_title('AP shift=4ms')
ax[2, 1].pcolormesh(P1, P2, ma.masked_invalid(error_APshift_2).T / np.max(ma.masked_invalid(error_APshift_2)))
ax[2, 1].plot(optimum[0], optimum[1], 'xk')
ax[2, 1].set_title('AP shift=2ms')
cbar = fig.colorbar(im, ax=ax.ravel().tolist())
for a in ax.ravel().tolist():
    a.set_xlabel('gmax na')
    a.set_ylabel('gmax k')
pl.savefig(save_dir+'/'+new_name+'/fitness_landscape.png')
#pl.show()


P1, P2 = np.meshgrid(p1_range, p2_range)
fig, ax = pl.subplots(3, 2, sharex=True, sharey=True)
im = ax[0, 0].contour(P1, P2, error_rms.T / np.max(error_rms))
ax[0, 0].plot(optimum[0], optimum[1], 'xk')
ax[0, 0].set_title('RMS(V, V*)')
ax[0, 1].contour(P1, P2, ma.masked_invalid(error_APamp).T / np.max(ma.masked_invalid(error_APamp)))
ax[0, 1].plot(optimum[0], optimum[1], 'xk')
ax[0, 1].set_title('AP amplitude')
ax[1, 0].contour(P1, P2, ma.masked_invalid(error_APwidth).T / np.max(ma.masked_invalid(error_APwidth)))
ax[1, 0].plot(optimum[0], optimum[1], 'xk')
ax[1, 0].set_title('AP width')
ax[1, 1].contour(P1, P2, ma.masked_invalid(error_APtime).T / np.max(ma.masked_invalid(error_APtime)))
ax[1, 1].plot(optimum[0], optimum[1], 'xk')
ax[1, 1].set_title('AP time')
im = ax[2, 0].contour(P1, P2, ma.masked_invalid(error_APshift_4).T / np.max(ma.masked_invalid(error_APshift_4)))
ax[2, 0].plot(optimum[0], optimum[1], 'xk')
ax[2, 0].set_title('AP shift=4ms')
ax[2, 1].contour(P1, P2, ma.masked_invalid(error_APshift_2).T / np.max(ma.masked_invalid(error_APshift_2)))
ax[2, 1].plot(optimum[0], optimum[1], 'xk')
ax[2, 1].set_title('AP shift=2ms')
cbar = fig.colorbar(im, ax=ax.ravel().tolist())
for a in ax.ravel().tolist():
    a.set_xlabel('gmax na')
    a.set_ylabel('gmax k')
pl.savefig(save_dir+'/'+new_name+'/fitness_landscape_contour.png')
pl.show()


with open(save_dir+'/error_rms.npy', 'w') as f:
    np.save(f, error_rms)
with open(save_dir + '/error_APamp.npy', 'w') as f:
    np.save(f, error_APamp)
with open(save_dir + '/error_APwidth.npy', 'w') as f:
    np.save(f, error_APwidth)
with open(save_dir + '/error_APtime.npy', 'w') as f:
    np.save(f, error_APtime)
with open(save_dir + '/error_APshift_4.npy', 'w') as f:
    np.save(f, error_APshift_4)
with open(save_dir + '/error_APshift_2.npy', 'w') as f:
    np.save(f, error_APshift_2)
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import os
from optimization.fitfuns import *
import pandas as pd
import json
import numpy.ma as ma
from optimization.fitnesslandscape import *
from optimization import errfuns

save_dir = '../../results/fitnesslandscapes/modellandscape/gna_gk/'
new_folder = 'fitfuns/stefans_fun'
fitfun = 'stefans_fun'
errfun_name = 'rms'
order = 1
optimum = [0.12, 0.036]
shift = 4
threshold = -30
window_before = 5
window_after = 20
bins_v = 100
bins_dvdt = 100
penalty = 0

with open(save_dir+'/chunk_size.txt', 'r') as f:
    chunk_size = int(f.read())
p1_range = np.loadtxt(save_dir + '/p1_range.txt')
p2_range = np.loadtxt(save_dir + '/p2_range.txt')
with open(save_dir + '/dirs.json', 'r') as f:
    dirs = json.load(f)
data = pd.read_csv(dirs['data_dir'])

errfun = getattr(errfuns, errfun_name)

# compute values for experimental data
dt = data.t.values[1] - data.t.values[0]
dvdt = np.concatenate((np.array([(data.v[1] - data.v[0]) / dt]), np.diff(data.v) / dt))

args = {'shift': shift, 'window_before': window_before, 'window_after': window_after, 'APtime': None,
        'threshold': threshold,
        'v_min': np.min(data.v), 'v_max': np.max(data.v), 'dvdt_min': np.min(dvdt), 'dvdt_max': np.max(dvdt),
        'bins_v': bins_v, 'bins_dvdt': bins_dvdt,
        'penalty': penalty,
        'v_mean': np.mean(data.v)}
v_data = get_v(np.array(data.v), data.t, data.i, args)
APtime_data = get_APtime(np.array(data.v), np.array(data.t), np.array(data.i), args)
args['APtime'] = APtime_data
APamp_data = get_APamp(np.array(data.v), np.array(data.t), np.array(data.i), args)
APwidth_data = get_APwidth(np.array(data.v), np.array(data.t), np.array(data.i), args)
AP_window_data = shifted_AP(np.array(data.v), np.array(data.t), np.array(data.i), args)
APmax_data = np.max(np.array(data.v))
phasehist_data = phase_hist(np.array(data.v), np.array(data.t), np.array(data.i), args)
has_1AP_data = penalize_not1AP(np.array(data.v), np.array(data.t), np.array(data.i), args)
vrest_data = get_vrest(np.array(data.v), np.array(data.t), np.array(data.i), args)
args['v_rest'] = vrest_data
i_inj_start = np.where(np.array(data.i) > 0)[0][0]
time_i_inj2AP = APtime_data - i_inj_start
v_half2AP_data = v_data[(i_inj_start + time_i_inj2AP / 2) * dt]
args['v_half2AP'] = v_half2AP_data

# compute fitness landscape
n_chunks_p1 = len(p1_range) / chunk_size
n_chunks_p2 = len(p2_range) / chunk_size
n_chunks_p1 = int(n_chunks_p1)
n_chunks_p2 = int(n_chunks_p2)
error = np.zeros((len(p1_range), len(p2_range)))
has_1AP_mat = np.zeros((len(p1_range), len(p2_range)))

for c1 in range(n_chunks_p1):
    for c2 in range(n_chunks_p2):
        with open(save_dir + '/modellandscape'+str(c1)+'_'+str(c2)+ '.npy', 'r') as f:
            modellandscape = np.load(f)

        for i in range(np.shape(modellandscape)[0]):
            for j in range(np.shape(modellandscape)[1]):
                has_1AP = penalize_not1AP(modellandscape[i, j], data.t, data.i, args) == 0
                has_1AP_mat[i + c1*chunk_size, j+c2*chunk_size] = has_1AP

                if fitfun == 'v_trace':
                    model_data = get_v(modellandscape[i, j], data.t, data.i, args)
                    exp_data = v_data
                elif fitfun == 'APshift':
                    model_data = shifted_AP(modellandscape[i, j], data.t, data.i, args)
                    exp_data = AP_window_data
                elif fitfun == 'APamp':
                    model_data = get_APamp(modellandscape[i, j], data.t, data.i, args)
                    exp_data = APamp_data
                elif fitfun == 'APwidth':
                    model_data = get_APwidth(modellandscape[i, j], data.t, data.i, args)
                    exp_data = APwidth_data
                elif fitfun == 'APtime':
                    model_data = get_APtime(modellandscape[i, j], data.t, data.i, args)
                    exp_data = APtime_data
                elif fitfun == 'phasehist':
                    model_data = phase_hist(modellandscape[i, j], data.t, data.i, args)
                    exp_data = phasehist_data
                elif fitfun == 'penalize_not1AP':
                    model_data = penalize_not1AP(modellandscape[i, j], data.t, data.i, args)
                    exp_data = has_1AP_data
                elif fitfun == 'v_rest':
                    model_data = get_vrest(modellandscape[i, j], data.t, data.i, args)
                    exp_data = vrest_data
                else:
                    model_data = None
                    exp_data = None

                if model_data is None:
                    error[i + c1*chunk_size, j+c2*chunk_size] = None
                else:
                    error[i + c1*chunk_size, j+c2*chunk_size] = errfun(model_data, exp_data)

                if fitfun == 'stefans_fun':
                    error[i + c1 * chunk_size, j + c2 * chunk_size] = stefans_fun(modellandscape[i, j],
                                                                                  data.t, data.i, args)


if not os.path.exists(save_dir + '/' + new_folder + '/'):
    os.makedirs(save_dir + '/' + new_folder + '/')
with open(save_dir + '/' + new_folder + '/error.npy', 'w') as f:
    np.save(f, error)
with open(save_dir + '/' + new_folder + '/has_1AP.npy', 'w') as f:
    np.save(f, has_1AP_mat)
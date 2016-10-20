import numpy as np
import matplotlib.pyplot as pl
from optimization.error_landscape_measures import *

save_dir = '../../../results/modellandscape/hhCell/gk_gl/'

with open(save_dir+'/error_rms.npy', 'r') as f:
    error_rms = np.load(f)
with open(save_dir + '/error_APamp.npy', 'r') as f:
    error_APamp = np.load(f)
with open(save_dir + '/error_APwidth.npy', 'r') as f:
    error_APwidth = np.load(f)
with open(save_dir + '/error_APtime.npy', 'r') as f:
    error_APtime = np.load(f)
with open(save_dir + '/error_APshift_4.npy', 'r') as f:
    error_APshift_4 = np.load(f)
with open(save_dir + '/error_APshift_2.npy', 'r') as f:
    error_APshift_2 = np.load(f)

order = 3

minima_x_rms = list()
minima_y_rms = list()
minima_xy_rms = list()
minima_x_APamp = list()
minima_y_APamp = list()
minima_xy_APamp = list()
minima_x_APwidth = list()
minima_y_APwidth = list()
minima_xy_APwidth = list()
minima_x_APtime = list()
minima_y_APtime = list()
minima_xy_APtime = list()
minima_x_APshift_4 = list()
minima_y_APshift_4 = list()
minima_xy_APshift_4 = list()
minima_x_APshift_2 = list()
minima_y_APshift_2 = list()
minima_xy_APshift_2 = list()

for i in range(np.shape(error_rms)[0]):
    minima_x_rms.extend([[i, y] for y in get_number_local_minima(error_rms[i, :], order=order)])
    minima_x_APamp.extend([[i, y] for y in get_number_local_minima(error_APamp[i, :], order=order)])
    minima_x_APwidth.extend([[i, y] for y in get_number_local_minima(error_APwidth[i, :], order=order)])
    minima_x_APtime.extend([[i, y] for y in get_number_local_minima(error_APtime[i, :], order=order)])
    minima_x_APshift_4.extend([[i, y] for y in get_number_local_minima(error_APshift_4[i, :], order=order)])
    minima_x_APshift_2.extend([[i, y] for y in get_number_local_minima(error_APshift_2[i, :], order=order)])

for i in range(np.shape(error_rms)[1]):
    minima_y_rms.extend([[x, i] for x in get_number_local_minima(error_rms[:, i], order=order)])
    minima_y_APamp.extend([[x, i] for x in get_number_local_minima(error_APamp[:, i], order=order)])
    minima_y_APwidth.extend([[x, i] for x in get_number_local_minima(error_APwidth[:, i], order=order)])
    minima_y_APtime.extend([[x, i] for x in get_number_local_minima(error_APtime[:, i], order=order)])
    minima_y_APshift_4.extend([[x, i] for x in get_number_local_minima(error_APshift_4[:, i], order=order)])
    minima_y_APshift_2.extend([[x, i] for x in get_number_local_minima(error_APshift_2[:, i], order=order)])


# problem with several dimensions: if several points at the trough minima can be overlooked
for minimum in minima_x_rms:
    if minimum in minima_y_rms:
        minima_xy_rms.append(minimum)
for minimum in minima_x_APamp:
    if minimum in minima_y_APamp:
        minima_xy_APamp.append(minimum)
for minimum in minima_x_APwidth:
    if minimum in minima_y_APwidth:
        minima_xy_APwidth.append(minimum)
for minimum in minima_x_APtime:
    if minimum in minima_y_APtime:
        minima_xy_APtime.append(minimum)
for minimum in minima_x_APshift_4:
    if minimum in minima_y_APshift_4:
        minima_xy_APshift_4.append(minimum)
for minimum in minima_x_APshift_2:
    if minimum in minima_y_APshift_2:
        minima_xy_APshift_2.append(minimum)


pl.figure()
w = 0.5
pl.bar(0-w/2, len(minima_xy_rms), w, color='k')
pl.bar(1-w/2, len(minima_xy_APamp), w, color='k')
pl.bar(2-w/2, len(minima_xy_APwidth), w, color='k')
pl.bar(3-w/2, len(minima_xy_APtime), w, color='k')
pl.bar(4-w/2, len(minima_xy_APshift_4), w, color='k')
pl.bar(5-w/2, len(minima_xy_APshift_2), w, color='k')
pl.xticks(range(6), ['RMS', 'AP amp', 'AP width', 'AP time', 'AP \nshift=4ms', 'AP \nshift=2ms'])
pl.savefig(save_dir+'n_minima.png')
pl.show()
import os
import numpy as np
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
pl.style.use('paper')


save_dir_characteristics = '../plots/spike_characteristics/distributions/rat'
save_dir_IV_characteristics = '../plots/IV/characteristics/'

save_dir_plots = os.path.join(save_dir_characteristics)

if not os.path.exists(save_dir_plots):
    os.makedirs(save_dir_plots)

characteristics = np.load(os.path.join(save_dir_characteristics, 'return_characteristics.npy'))
characteristics_mat = np.load(os.path.join(save_dir_characteristics, 'characteristics_mat.npy'))
cell_ids_characteristics = np.load(os.path.join(save_dir_characteristics, 'cell_ids.npy'))

ISI_1st = np.load(os.path.join(save_dir_IV_characteristics, 'ISI_1st.npy'))
lag_1st_AP = np.load(os.path.join(save_dir_IV_characteristics, 'lag_1st_AP.npy'))
cell_ids_IV = np.load(os.path.join(save_dir_IV_characteristics, 'cell_ids.npy'))


cell_ids_both = np.intersect1d(cell_ids_characteristics, cell_ids_IV)
cell_ids_characteristics_idxs = np.array([np.where(cell_ids_characteristics == cell_id)[0] for cell_id in cell_ids_both])
cell_ids_IV_idxs = np.array([np.where(cell_ids_IV == cell_id)[0] for cell_id in cell_ids_both])

pl.figure()
pl.plot(ISI_1st,
        lag_1st_AP, 'ok')
pl.xlabel('1st ISI (ms)')
pl.ylabel('Lag 1st AP (ms)')

pl.figure()
pl.plot(characteristics_mat[cell_ids_characteristics_idxs, characteristics == 'DAP_width'],
        ISI_1st[cell_ids_IV_idxs], 'ok')
pl.xlabel('DAP width (ms)')
pl.ylabel('1st ISI (ms)')

pl.figure()
pl.plot(characteristics_mat[cell_ids_characteristics_idxs, characteristics == 'DAP_amp'],
        ISI_1st[cell_ids_IV_idxs], 'ok')
pl.xlabel('DAP amp (mV)')
pl.ylabel('1st ISI (ms)')

pl.figure()
pl.plot(characteristics_mat[cell_ids_characteristics_idxs, characteristics == 'DAP_width'],
        lag_1st_AP[cell_ids_IV_idxs], 'ok')
pl.xlabel('DAP width (ms)')
pl.ylabel('Lag 1st AP (ms)')

pl.figure()
pl.plot(characteristics_mat[cell_ids_characteristics_idxs, characteristics == 'DAP_amp'],
        lag_1st_AP[cell_ids_IV_idxs], 'ok')
pl.xlabel('DAP amp (mV)')
pl.ylabel('Lag 1st AP (ms)')

pl.figure()
pl.plot(characteristics_mat[cell_ids_characteristics_idxs, characteristics == 'height_3ms_after_AP'],
        lag_1st_AP[cell_ids_IV_idxs], 'ok')
pl.xlabel('DAP height (mV)')
pl.ylabel('Lag 1st AP (ms)')

pl.figure()
pl.plot(characteristics_mat[cell_ids_characteristics_idxs, characteristics == 'height_3ms_after_AP'],
        ISI_1st[cell_ids_IV_idxs], 'ok')
pl.xlabel('DAP height (mV)')
pl.ylabel('1st ISI (ms)')

fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(ISI_1st[cell_ids_IV_idxs].flatten(),
        characteristics_mat[cell_ids_characteristics_idxs, characteristics == 'DAP_amp'].flatten().astype(float),
        lag_1st_AP[cell_ids_IV_idxs].flatten(), 'ok', alpha=0.5, markersize=4.0)
ax.set_xlabel('1st ISI (ms)')
ax.set_ylabel('DAP amp (mV)')
ax.set_zlabel('Lag 1st AP (ms)')

pl.show()

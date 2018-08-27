from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, get_i_inj_from_function, \
    get_sweep_index_for_amp
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
from cell_characteristics.analyze_APs import get_AP_onset_idxs
import seaborn as sns
import pandas as pd
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    save_dir_img = '../plots/IV/latency_vs_ISI12/rat/summary'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    protocol = 'IV'
    cell_ids = get_cells_for_protocol(data_dir, protocol)
    animal = 'rat'
    AP_threshold = 0
    start_step = 250  # ms

    cell_ids_used = []
    latency = []
    ISI12 = []

    for cell_id in cell_ids:
        if not check_rat_or_gerbil(cell_id) == animal:
            continue

        # load data
        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                         return_sweep_idxs=True)
        t = t_mat[0, :]
        dt = t[1] - t[0]
        i_inj_mat = get_i_inj_from_function(protocol, sweep_idxs, t[-1], t[1]-t[0])

        # get latency and ISI1/2
        latency_cell = None
        for v in v_mat:
            AP_onsets = get_AP_onset_idxs(v, AP_threshold)

            if len(AP_onsets) >= 1:
                latency_cell = t[AP_onsets[0]] - start_step
                # print latency_cell
                # pl.figure()
                # pl.plot(t, v)
                # pl.show()
                break

        ISI12_cell = None
        for v in v_mat:
            AP_onsets = get_AP_onset_idxs(v, AP_threshold)

            if len(AP_onsets) >= 4:
                ISIs = np.diff(AP_onsets) * dt
                ISI12_cell = ISIs[0] / ISIs[1]
                # print ISI12_cell
                # pl.figure()
                # pl.plot(t, v)
                # pl.show()
                break

        if latency_cell is not None and ISI12_cell is not None:
            cell_ids_used.append(cell_id)
            latency.append(latency_cell)
            ISI12.append(ISI12_cell)

    # plot
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    np.save(os.path.join(save_dir_img, 'latency.npy'), latency)
    np.save(os.path.join(save_dir_img, 'ISI12.npy'), ISI12)
    np.save(os.path.join(save_dir_img, 'cell_ids.npy'), cell_ids_used)

    data = pd.DataFrame(np.array([latency, ISI12]).T, columns=['Latency', 'ISI1/2'])
    jp = sns.jointplot('Latency', 'ISI1/2', data=data, stat_func=None, color='0.5')
    jp.fig.set_size_inches(6.4, 4.8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'latency_vs_ISI12.png'))

    pl.show()
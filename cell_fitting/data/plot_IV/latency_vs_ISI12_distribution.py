from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, get_i_inj_from_function, \
    get_amp_for_sweep_index,get_i_inj_standard_params
from cell_fitting.data.plot_IV import check_v_at_i_inj_0_is_at_right_sweep_idx
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
from cell_characteristics.analyze_APs import get_AP_onset_idxs
import seaborn as sns
import pandas as pd
from cell_fitting.data import check_cell_has_DAP
from cell_characteristics.analyze_step_current_data import get_latency_to_first_spike, get_ISI12
from cell_characteristics import to_idx
pl.style.use('paper')


if __name__ == '__main__':
    # parameters
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    protocol = 'IV'
    animal = 'rat'
    AP_threshold = -10
    save_dir_img = os.path.join('../plots/', protocol, 'latency_vs_ISI12', animal)

    # get cell_ids
    cell_ids = get_cells_for_protocol(data_dir, protocol)
    cell_ids = filter(lambda id: check_rat_or_gerbil(id) == animal, cell_ids)
    cell_ids = filter(lambda id: check_cell_has_DAP(id), cell_ids)

    cell_ids_used = []
    latencies = []
    ISI12s = []
    for cell_id in cell_ids:
        # load data
        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                         return_sweep_idxs=True)
        t = t_mat[0, :]
        dt = t[1] - t[0]
        i_inj_mat = get_i_inj_from_function(protocol, sweep_idxs, t[-1], t[1]-t[0])
        params = get_i_inj_standard_params(protocol, sweep_idxs=sweep_idxs)
        amps = params['step_amp']
        start_step = params['start_step']
        end_step = params['end_step']

        try:
            check_v_at_i_inj_0_is_at_right_sweep_idx(v_mat, i_inj_mat, to_idx(start_step, dt), to_idx(end_step, dt))
        except AssertionError:
            continue
        if get_amp_for_sweep_index(sweep_idxs[-1], protocol) <= 0:  # not enough positive amplitudes tested
            continue

        # get latency and ISI1/2
        latency_cell = None
        for v in v_mat:
            AP_onsets = get_AP_onset_idxs(v, AP_threshold)
            latency_cell = get_latency_to_first_spike(v, t, AP_onsets, start_step, end_step)
            if latency_cell is not None:
               break

        ISI12_cell = None
        for v in v_mat:
            AP_onsets = get_AP_onset_idxs(v, AP_threshold)

            ISI12_cell = get_ISI12(v, t, AP_onsets, start_step, end_step)
            if ISI12_cell is not None:
                break

        if latency_cell is not None and ISI12_cell is not None:
            cell_ids_used.append(cell_id)
            latencies.append(latency_cell)
            ISI12s.append(ISI12_cell)
        else:
            print 'Bad cell: ', cell_id

    # plot
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    np.save(os.path.join(save_dir_img, 'latency.npy'), latencies)
    np.save(os.path.join(save_dir_img, 'ISI12.npy'), ISI12s)
    np.save(os.path.join(save_dir_img, 'cell_ids.npy'), cell_ids_used)

    data = pd.DataFrame(np.array([latencies, ISI12s]).T, columns=['Latency', 'ISI1/2'])
    jp = sns.jointplot('Latency', 'ISI1/2', data=data, stat_func=None, color='0.5')
    jp.fig.set_size_inches(6.4, 4.8)
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'latency_vs_ISI12.png'))

    pl.show()
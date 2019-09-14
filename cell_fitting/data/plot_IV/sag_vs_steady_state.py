import matplotlib.pyplot as pl
import numpy as np
import os
import json
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function, get_cells_for_protocol, shift_v_rest
from cell_fitting.optimization.evaluation.plot_IV.potential_sag_vs_steady_state import compute_v_sag_and_steady_state
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
from cell_fitting.optimization.evaluation.plot_IV.potential_sag_vs_steady_state import plot_sag_vs_steady_state_on_ax
from cell_fitting.data import check_cell_has_DAP
pl.style.use('paper')


if __name__ == '__main__':

    # parameters
    #data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    data_dir = '/media/cfischer/TOSHIBA EXT/Sicherung_2018_05_19/Phd/DAP-Project/cell_data/raw_data'
    AP_threshold = -30
    v_shift = -16
    animal = 'rat'
    protocol = 'IV'
    save_dir = os.path.join('../plots', protocol, 'sag', animal)

    # get cell_ids
    cell_ids = get_cells_for_protocol(data_dir, protocol)
    cell_ids = filter(lambda id: check_rat_or_gerbil(id) == animal, cell_ids)
    cell_ids = filter(lambda id: check_cell_has_DAP(id), cell_ids)

    #cell_ids = ['2015_05_26d', '2015_06_08a', '2015_06_09f', '2015_06_19i', '2015_08_10g', '2015_08_26b']
    cell_ids = ['2015_08_26b']

    for cell_id in cell_ids:
        # read data
        v_mat_data, t_mat_data, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), 'IV',
                                                                   return_sweep_idxs=True)
        v_mat_data = [shift_v_rest(v, v_shift) for v in v_mat_data]
        i_inj_mat = get_i_inj_from_function('IV', sweep_idxs, t_mat_data[0][-1], t_mat_data[0][1]-t_mat_data[0][0])

        # compute amplitudes
        start_step = np.nonzero(i_inj_mat[0])[0][0]
        end_step = np.nonzero(i_inj_mat[0])[0][-1] + 1
        amps = np.array([i_inj[start_step] for i_inj in i_inj_mat])

        v_sags, v_steady_states, amps_subtheshold = compute_v_sag_and_steady_state(v_mat_data, amps, AP_threshold,
                                                                                   start_step, end_step)

        # save
        max_amp = 0.15
        amps_subtheshold_range = np.array(amps_subtheshold) < max_amp + 0.05
        amps_subtheshold = np.array(amps_subtheshold)[amps_subtheshold_range]
        v_steady_states = np.array(v_steady_states)[amps_subtheshold_range]
        v_sags = np.array(v_sags)[amps_subtheshold_range]

        sag_dict = dict(amps_subtheshold=list(amps_subtheshold), v_steady_states=list(v_steady_states),
                        v_sags=list(v_sags))
        save_dir_dict = os.path.join('../plots/', 'IV', 'sag', cell_id)
        if not os.path.exists(save_dir_dict):
            os.makedirs(save_dir_dict)
        with open(os.path.join(save_dir_dict, 'sag_dict.json'), 'w') as f:
            json.dump(sag_dict, f)

        # plot
        save_dir_img = os.path.join(save_dir, cell_id)
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        fig, ax = pl.subplots()
        plot_sag_vs_steady_state_on_ax(ax, amps_subtheshold, v_steady_states, v_sags)
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'sag_vs_steady_state.png'))
        #pl.show()
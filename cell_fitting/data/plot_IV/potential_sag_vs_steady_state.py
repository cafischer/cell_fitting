import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function, get_cells_for_protocol
from cell_fitting.optimization.evaluation.plot_IV.potential_sag_vs_steady_state import compute_v_sag_and_steady_state
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
pl.style.use('paper')


if __name__ == '__main__':

    # parameters
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    AP_threshold = -30
    v_shift = -16
    animal = 'rat'
    save_dir = os.path.join('../plots/', 'plot_IV', 'sag_vs_steady_state', animal)

    cells = get_cells_for_protocol(data_dir, 'plot_IV')
    cells = ['2015_05_26d', '2015_06_08a', '2015_06_09f', '2015_06_19i', '2015_08_10g', '2015_08_26b']

    for cell_id in cells:
        if not '2015' in cell_id:
            continue
        if not check_rat_or_gerbil(cell_id) == animal:
            continue

        # read data
        v_mat_data, t_mat_data, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), 'plot_IV',
                                                                   return_sweep_idxs=True)
        i_inj_mat = get_i_inj_from_function('plot_IV', sweep_idxs, t_mat_data[0][-1], t_mat_data[0][1]-t_mat_data[0][0])

        # compute amplitudes
        start_step = np.nonzero(i_inj_mat[0])[0][0]
        end_step = np.nonzero(i_inj_mat[0])[0][-1] + 1
        amps = np.array([i_inj[start_step] for i_inj in i_inj_mat])

        v_sags, v_steady_states, amps_subtheshold = compute_v_sag_and_steady_state(v_mat_data, amps, AP_threshold,
                                                                                   start_step, end_step)

        # plot
        save_dir_img = os.path.join(save_dir, cell_id)
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        max_amp = 0.15
        amps_subtheshold = np.array(amps_subtheshold)
        amps_subtheshold_range = amps_subtheshold < max_amp + 0.05

        pl.figure()
        pl.plot(amps_subtheshold[amps_subtheshold_range], np.array(v_steady_states)[amps_subtheshold_range], linestyle='-',
                marker='o', c='0.0', label='Steady State')
        pl.plot(amps_subtheshold[amps_subtheshold_range], np.array(v_sags)[amps_subtheshold_range], linestyle='-',
                marker='o', c='0.5', label='Sag')
        pl.xlabel('Current (nA)')
        pl.ylabel('Membrane Potential (mV)')
        pl.legend(loc='upper left')
        pl.xticks(np.arange(-0.15, max_amp+0.05, 0.05))
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'sag_steady_state.png'))
        pl.show()
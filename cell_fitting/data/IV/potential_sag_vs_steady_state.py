import matplotlib.pyplot as pl
import numpy as np
import os
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj, get_cells_for_protocol
from cell_fitting.new_optimization.evaluation.IV.potential_sag_vs_steady_state import compute_v_sag_and_steady_state
pl.style.use('paper')


if __name__ == '__main__':

    # parameters
    mechanism_dir = '../../../model/channels/vavoulis'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    save_dir = os.path.join('../plots/', 'IV', 'sag_vs_steady_state')
    AP_threshold = -30
    v_shift = -16

    cells = get_cells_for_protocol(data_dir, 'IV')

    for cell in cells:
        if not '2015' in cell:
            continue

        # read data
        v_mat_data, t_mat_data, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell+'.dat'), 'IV',
                                                                   return_sweep_idxs=True)
        i_inj_mat = get_i_inj('IV', sweep_idxs)

        # compute amplitudes
        start_step = np.nonzero(i_inj_mat[0])[0][0]
        end_step = np.nonzero(i_inj_mat[0])[0][-1] + 1
        amps = np.array([i_inj[start_step] for i_inj in i_inj_mat])

        v_sags, v_steady_states, amps_subtheshold = compute_v_sag_and_steady_state(v_mat_data, amps, AP_threshold,
                                                                                   start_step, end_step)

        # plot
        save_dir_img = os.path.join(save_dir, cell)
        if not os.path.exists(save_dir_img):
            os.makedirs(save_dir_img)

        pl.figure()
        pl.plot(amps_subtheshold, v_steady_states, linestyle='-', marker='o', c='0.0', label='Steady State')
        pl.plot(amps_subtheshold, v_sags, linestyle='-', marker='o', c='0.5', label='Sag')
        pl.xlabel('Current (nA)')
        pl.ylabel('Membrane Potential (mV)')
        pl.legend(loc='upper left')
        pl.tight_layout()
        pl.savefig(os.path.join(save_dir_img, 'sag_steady_state.png'))
        #pl.show()
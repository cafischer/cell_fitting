import matplotlib.pyplot as pl
import numpy as np
import os
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, get_i_inj


if __name__ == '__main__':

    # parameters
    save_dir = './plots/rampIV'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    protocol = 'rampIV'
    v_rest_shift = -16
    AP_threshold = -30
    cells = get_cells_for_protocol(data_dir, protocol)

    for cell in cells:
        if not '2015' in cell:
            continue

        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell+'.dat'), protocol,
                                                         return_sweep_idxs=True)
        i_inj_mat = get_i_inj(protocol, sweep_idxs)
        t = t_mat[0, :]

        for v in v_mat:
            AP_onset_idxs = get_AP_onset_idxs(v, AP_threshold)
            if len(AP_onset_idxs >= 1):
                continue

        # plot
        save_dir_fig = os.path.join(save_dir, cell)
        if not os.path.exists(save_dir_fig):
            os.makedirs(save_dir_fig)

        pl.figure()
        pl.plot(t, v, 'k', label='Exp. Data')
        pl.xlabel('Time (ms)', fontsize=16)
        pl.ylabel('Membrane Potential (mV)', fontsize=16)
        pl.legend(loc='lower right', fontsize=16)
        pl.savefig(os.path.join(save_dir_fig, 'rampIV.png'))
        #pl.show()
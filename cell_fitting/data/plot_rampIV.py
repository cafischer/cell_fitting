import matplotlib.pyplot as pl
import numpy as np
import os
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, get_i_inj_from_function
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
pl.style.use('paper')


def find_current_threshold_data(v_mat, i_inj_mat, AP_threshold):
    current_threshold = None
    idx = None
    for idx, (v, i_inj) in enumerate(zip(v_mat, i_inj_mat)):
        AP_onset_idxs = get_AP_onset_idxs(v, AP_threshold)
        if len(AP_onset_idxs) >= 1:
            current_threshold = np.max(i_inj)
            break
    return current_threshold, idx


if __name__ == '__main__':

    # parameters
    save_dir = './plots/rampIV/rat'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    protocol = 'rampIV'
    v_rest_shift = -16
    AP_threshold = -10
    #cells = get_cells_for_protocol(data_dir, protocol)
    cells = ['2014_07_10b', '2014_07_02a', '2014_07_03a', '2017_07_08d', '2014_07_09c', '2014_07_09e', '2014_07_09f', '2014_07_10d']
    animal = 'rat'

    for cell_id in cells:
        #if not '2015' in cell_id:
        #    continue
        if not check_rat_or_gerbil(cell_id) == animal:
            continue

        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                         return_sweep_idxs=True)
        i_inj_mat = get_i_inj_from_function(protocol, sweep_idxs, t_mat[0][-1], t_mat[0][1]-t_mat[0][0])
        t = t_mat[0, :]

        current_threshold, idx = find_current_threshold_data(v_mat, i_inj_mat, AP_threshold)

        if current_threshold is not None:
            # plot
            save_dir_img = os.path.join(save_dir, cell_id)
            if not os.path.exists(save_dir_img):
                os.makedirs(save_dir_img)

            np.savetxt(os.path.join(save_dir_img, 'current_threshold.txt'), np.array([current_threshold]))

            pl.figure()
            pl.plot(t, v_mat[idx], 'k', label='Exp. Data')
            pl.xlabel('Time (ms)')
            pl.ylabel('Membrane Potential (mV)')
            #pl.legend()
            pl.tight_layout()
            pl.savefig(os.path.join(save_dir_img, 'v.png'))
            pl.show()
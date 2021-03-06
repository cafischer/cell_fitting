import matplotlib.pyplot as pl
import numpy as np
import os
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from cell_fitting.read_heka import get_v_and_t_from_heka, get_cells_for_protocol, get_i_inj_from_function
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
from cell_fitting.data import check_cell_has_DAP
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
    save_dir = '../plots/rampIV/rat'
    #data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    data_dir = '/media/cfischer/TOSHIBA EXT/Sicherung_2018_05_19/Phd/DAP-Project/cell_data/raw_data'
    protocol = 'rampIV'
    v_rest_shift = -16
    AP_threshold = -10
    animal = 'rat'

    # get cell_ids
    cell_ids = get_cells_for_protocol(data_dir, protocol)
    cell_ids = filter(lambda id: check_rat_or_gerbil(id) == animal, cell_ids)
    cell_ids = filter(lambda id: check_cell_has_DAP(id), cell_ids)
    # cell_ids = ['2014_07_10b', '2014_07_02a', '2014_07_03a', '2017_07_08d', '2014_07_09c', '2014_07_09e', '2014_07_09f',
    #          '2014_07_10d']
    #cell_ids = ['2015_08_26b']

    current_thresholds = np.zeros(len(cell_ids))
    v_rest = np.zeros(len(cell_ids))

    for cell_idx, cell_id in enumerate(cell_ids):
        print cell_id

        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                         return_sweep_idxs=True)
        i_inj_mat = get_i_inj_from_function(protocol, sweep_idxs, t_mat[0][-1], t_mat[0][1]-t_mat[0][0])
        t = t_mat[0, :]

        v_rest[cell_idx] = np.mean(v_mat[0, :100])

        current_threshold, idx = find_current_threshold_data(v_mat, i_inj_mat, AP_threshold)

        if current_threshold is not None:
            # plot
            save_dir_img = os.path.join(save_dir, cell_id)
            if not os.path.exists(save_dir_img):
                os.makedirs(save_dir_img)

            np.savetxt(os.path.join(save_dir_img, 'current_threshold.txt'), np.array([current_threshold]))
            current_thresholds[cell_idx] = current_threshold

            # pl.figure()
            # pl.plot(t, v_mat[idx], 'k', label='Exp. Data')
            # pl.xlabel('Time (ms)')
            # pl.ylabel('Membrane Potential (mV)')
            # #pl.legend()
            # pl.tight_layout()
            # pl.savefig(os.path.join(save_dir_img, 'v.png'))
            # pl.show()

            # fig, ax = pl.subplots(2, 1, sharex=True)
            # ax[0].plot(t, v_mat[idx], 'k')
            # ax[1].plot(t, i_inj_mat[idx], 'k')
            # pl.xlabel('Time (ms)', fontsize=16)
            # ax[0].set_ylabel('Membrane \nPotential (mV)', fontsize=16)
            # ax[1].set_ylabel('Injected \nCurrent (nA)', fontsize=16)
            # pl.tight_layout()
            # pl.savefig(os.path.join(save_dir_img, 'v_i.png'))
            # pl.show()

            # import pandas as pd
            # data = pd.DataFrame(np.vstack((t, v_mat[idx], i_inj_mat[idx])).T, columns=['t', 'v', 'i'])
            # data.to_csv(os.path.join(save_dir_img, cell_id+'_rampIV.csv'))

    print np.mean(v_rest)
    print np.std(v_rest)

    pl.figure()
    pl.hist(current_thresholds)
    pl.show()

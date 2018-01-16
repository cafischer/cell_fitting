import numpy as np
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
import os
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function
from cell_characteristics.analyze_APs import get_spike_characteristics, get_AP_onset_idxs
from cell_fitting.data import shift_v_rest


if __name__ == '__main__':

    save_dir = '../plots/spike_characteristics/distributions/rat'
    data_dir = '/home/cf/Phd/DAP-Project/cell_data/raw_data'
    file_names = os.listdir(data_dir)
    cell_ids = [f_n[:-4] for f_n in file_names][0:]
    #cell_ids = ['2015_08_26b']
    protocol = 'rampIV'
    animal = 'rat'

    v_rest_shift = -16
    dt = 0.01
    AP_threshold = -30  # mV
    AP_interval = 2.5  # ms (also used as interval for fAHP)
    AP_width_before_onset = 2  # ms
    fAHP_interval = 4.0
    DAP_interval = 10  # ms
    order_fAHP_min = 1.0  # ms (how many points to consider for the minimum)
    order_DAP_max = 1.0  # ms (how many points to consider for the minimum)
    min_dist_to_DAP_max = 0.5  # ms
    k_splines = 3
    s_splines = None
    return_characteristics = ['AP_amp', 'AP_width', 'fAHP_amp', 'DAP_amp', 'DAP_deflection', 'DAP_width', 'DAP_time',
                              'fAHP2DAP_time']

    cell_id_list = []
    spike_characteristics_list = []
    v_list = []
    count_cells_rampIV = 0
    for cell_id in cell_ids:
        # check right animal
        if not check_rat_or_gerbil(cell_id) == animal:
            continue

        # find trace with AP
        try:
            v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                             return_sweep_idxs=True)
            count_cells_rampIV += 1
            for i, s in enumerate(sweep_idxs):
                onset_idxs = get_AP_onset_idxs(v_mat[i, :], AP_threshold)
                if len(onset_idxs) == 1 and onset_idxs * dt < 12.5 and len(v_mat[i, :]) == 16200:
                    v = shift_v_rest(v_mat[i, :], v_rest_shift)
                    t = t_mat[i, :]
                    i_inj = get_i_inj_from_function(protocol, sweep_idxs, t[-1], t[1]-t[0])[i, :]
                    dt_d = t[1] - t[0]
                    assert dt == dt_d
                    start_i_inj = np.where(np.diff(np.abs(i_inj)) > 0)[0][0] + 1

                    # get spike characteristics
                    std_end = min(10, start_i_inj * dt)
                    std_idx_times = (0, std_end)
                    v_rest = np.mean(v[0:start_i_inj])
                    spike_characteristics = np.array(get_spike_characteristics(v, t, return_characteristics, v_rest,
                                                                               AP_threshold, AP_interval, AP_width_before_onset,
                                                                               fAHP_interval, std_idx_times, k_splines,
                                                                               s_splines, order_fAHP_min, DAP_interval,
                                                                               order_DAP_max, min_dist_to_DAP_max,
                                                                               check=False))
                    if None not in spike_characteristics and np.all(spike_characteristics[:-2] >= 0):
                        cell_id_list.append(cell_id)
                        spike_characteristics_list.append(spike_characteristics)
                        v_list.append(v)
                    break
        except KeyError:
            continue
    characteristics_mat = np.vstack(spike_characteristics_list)  # candidates vs characteristics
    AP_matrix = np.vstack(v_list)  # candidates vs t

    print 'cells with rampIV: ' + str(count_cells_rampIV)
    print 'cells with DAP: ' + str(len(AP_matrix))

    # save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'return_characteristics.npy'), return_characteristics)
    np.save(os.path.join(save_dir, 'characteristics_mat.npy'), characteristics_mat)
    np.save(os.path.join(save_dir, 'AP_mat.npy'), AP_matrix)
    np.save(os.path.join(save_dir, 't.npy'), t)
    np.save(os.path.join(save_dir, 'cell_ids.npy'), cell_id_list)
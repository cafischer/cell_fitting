import numpy as np
from cell_fitting.data.divide_rat_gerbil_cells import check_rat_or_gerbil
import os
from cell_fitting.read_heka import get_v_and_t_from_heka, get_i_inj_from_function, shift_v_rest, get_cells_for_protocol
from cell_characteristics.analyze_APs import get_spike_characteristics, get_AP_onset_idxs
from cell_fitting.optimization.evaluation import get_spike_characteristics_dict


if __name__ == '__main__':
    # data_dir = '/home/cfischer/Phd/DAP-Project/cell_data/raw_data'
    data_dir = '/media/cfischer/TOSHIBA EXT/Sicherung_2018_05_19/Phd/DAP-Project/cell_data/raw_data'
    protocol = 'rampIV'
    animal = 'rat'
    save_dir = os.path.join('../plots/spike_characteristics', animal)

    # get cell_ids
    cell_ids = get_cells_for_protocol(data_dir, protocol)
    cell_ids = filter(lambda id: check_rat_or_gerbil(id) == animal, cell_ids)

    v_rest_shift = -16
    dt = 0.01
    spike_characteristics_dict = get_spike_characteristics_dict(for_data=True)
    return_characteristics = ['AP_amp', 'AP_width', 'fAHP_amp', 'DAP_amp', 'DAP_deflection', 'DAP_width', 'DAP_time',
                              'fAHP2DAP_time', 'height_3ms_after_AP']
    characteristics_with_range = ['AP_amp', 'AP_width', 'fAHP_amp', 'DAP_amp', 'DAP_deflection', 'DAP_width', 'DAP_time']
    characteristics_valid_ranges = [(50, 150), (0.1, 2.0), (0, 40), (0, 40), (0, 20), (0, 70), (0, 20)]

    cell_ids_with_DAP = []
    spike_characteristics_with_DAP = []
    v_with_DAP = []
    ramp_amps = []
    count_cells_no_DAP = 0
    for cell_id in cell_ids:
        # find trace with AP
        v_mat, t_mat, sweep_idxs = get_v_and_t_from_heka(os.path.join(data_dir, cell_id + '.dat'), protocol,
                                                             return_sweep_idxs=True)
        for i, s in enumerate(sweep_idxs):
            onset_idxs = get_AP_onset_idxs(v_mat[i, :], spike_characteristics_dict['AP_threshold'])
            if len(onset_idxs) == 1 and onset_idxs * dt < 12.5 and len(v_mat[i, :]) == 16200:
                v = shift_v_rest(v_mat[i, :], v_rest_shift)
                t = t_mat[i, :]
                i_inj = get_i_inj_from_function(protocol, sweep_idxs, t[-1], t[1]-t[0])[i, :]
                dt_d = t[1] - t[0]
                assert dt == dt_d
                start_i_inj = np.where(np.diff(np.abs(i_inj)) > 0)[0][0] + 1

                # get spike characteristics
                std_idx_times = (0, min(1, start_i_inj * dt))
                v_rest = np.mean(v[0:start_i_inj])
                spike_characteristics = np.array(get_spike_characteristics(v, t, return_characteristics, v_rest,
                                                                           check=False, std_idx_times=std_idx_times,
                                                                           **spike_characteristics_dict))

                # check characteristics in valid range
                all_valid = True
                for characteristic, valid_range in zip(characteristics_with_range, characteristics_valid_ranges):
                    characteristic_idx = np.where(characteristic == np.array(return_characteristics))[0][0]
                    if spike_characteristics[characteristic_idx] is None \
                            or not valid_range[0] <= spike_characteristics[characteristic_idx] <= valid_range[1]:
                        all_valid = False
                        if spike_characteristics[characteristic_idx] > valid_range[1]:
                            print spike_characteristics[characteristic_idx], valid_range[1]
                        break

                if all_valid is True:
                    cell_ids_with_DAP.append(cell_id)
                    spike_characteristics_with_DAP.append(spike_characteristics)
                    v_with_DAP.append(v)
                    ramp_amps.append(np.max(i_inj))

                if spike_characteristics[4] is None:
                    count_cells_no_DAP += 1
                    # # TODO
                    # spike_characteristics = np.array(get_spike_characteristics(v, t, return_characteristics, v_rest,
                    #                                                            check=False,
                    #                                                            std_idx_times=std_idx_times,
                    #                                                            **spike_characteristics_dict))
                break

    characteristics_mat = np.vstack(spike_characteristics_with_DAP)  # candidates vs characteristics
    AP_matrix = np.vstack(v_with_DAP)  # candidates vs t

    print 'cells with rampIV: ' + str(len(cell_ids))
    print 'cells with DAP: ' + str(len(AP_matrix))
    print 'cells with no DAP: ' + str(count_cells_no_DAP)

    # save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'return_characteristics.npy'), return_characteristics)
    np.save(os.path.join(save_dir, 'characteristics_mat.npy'), characteristics_mat)
    np.save(os.path.join(save_dir, 'AP_mat.npy'), AP_matrix)
    np.save(os.path.join(save_dir, 't.npy'), t)
    np.save(os.path.join(save_dir, 'cell_ids.npy'), cell_ids_with_DAP)
    np.save(os.path.join(save_dir, 'ramp_amps.npy'), ramp_amps)
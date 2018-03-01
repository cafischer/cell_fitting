from cell_fitting.read_heka import *
import pandas as pd


def read_data(data_dir, cell_id, protocol, sweep_idx, v_rest_shift=0, file_type='dat', return_discontinuities=False):
    if file_type == 'csv':
        if return_discontinuities:
            raise ValueError('Not possible to return discontinuities for csv!')
        else:
            v, t, i_inj = read_data_from_csv(data_dir, cell_id, protocol, sweep_idx)
    elif file_type == 'dat':
        if return_discontinuities:
            v, t, i_inj, discontinuities = read_data_from_dat(data_dir, cell_id, protocol, sweep_idx, True)
            return {'v': v, 't': t, 'i_inj': i_inj}, discontinuities
        else:
            v, t, i_inj = read_data_from_dat(data_dir, cell_id, protocol, sweep_idx)
    else:
        raise ValueError('Unknown file type!')
    v = shift_v_rest(v, v_rest_shift)
    return {'v': v, 't': t, 'i_inj': i_inj}


def read_data_from_csv(data_dir, cell_id, protocol, sweep_idx):
    data = pd.read_csv(os.path.join(data_dir, cell_id, protocol, str(sweep_idx)+'.csv'))
    return data.v.values, data.t.values, data.i.values


def read_data_from_dat(data_dir, cell_id, protocol, sweep_idx, return_discontinuities=False):
    v_mat, t_mat = get_v_and_t_from_heka(os.path.join(data_dir, cell_id+'.dat'), protocol, sweep_idxs=[sweep_idx])
    i_inj_mat = get_i_inj_from_function(protocol, [sweep_idx], t_mat[0][-1], t_mat[0][1]-t_mat[0][0])
    if return_discontinuities:
        i_inj_mat, discontinuities = get_i_inj_from_function(protocol,
                                        [sweep_idx], t_mat[0][-1], t_mat[0][1] - t_mat[0][0], True)
        return v_mat[0], t_mat[0], i_inj_mat[0], discontinuities
    return v_mat[0], t_mat[0], i_inj_mat[0]
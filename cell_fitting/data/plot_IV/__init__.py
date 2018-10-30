import numpy as np
import matplotlib.pyplot as pl
from cell_fitting.read_heka import get_sweep_index_for_amp


def get_index_i_inj_start_end(i_inj):
    nonzero = np.nonzero(i_inj)[0]
    if len(nonzero) <= 1:
        return None
    else:
        return nonzero[0], nonzero[-1]


def check_v_at_i_inj_0_is_at_right_sweep_idx(v_mat, i_inj_mat, step_start_idx, step_end_idx, acceptable_diff=0.5):
    sweep0 = get_sweep_index_for_amp(0, 'IV')
    if np.shape(v_mat)[0] > sweep0:  # trace exists
        v_rest_mean = np.mean(np.concatenate((v_mat[sweep0, :step_start_idx], v_mat[sweep0, step_end_idx:])))
        v_step_mean = np.mean(v_mat[sweep0, step_start_idx:step_end_idx])
        if np.abs(v_rest_mean - v_step_mean) >= acceptable_diff:
            # pl.figure()
            # pl.plot(v_mat[sweep0, :])
            # pl.show()
            # mV mean values should be roughly the same
            raise AssertionError('V at i_inj=0 is not ok!')
import pandas as pd
import numpy as np
from cell_fitting.read_heka.i_inj_functions import get_i_inj_double_ramp


def get_ramp3_times(delta_first=3, delta_ramp=2, n_times=10):
    return np.arange(delta_first, n_times * delta_ramp + delta_ramp, delta_ramp)


def get_i_inj_double_ramp_full(ramp_amp, ramp3_amps, ramp3_times, step_amps, len_step=250, baseline_amp=-0.05,
                               len_ramp=2, start_ramp1=20, start_step=222, len_step2ramp=15, tstop=500, dt=0.01):

    i_inj_mat = np.zeros((len(ramp3_amps), len(ramp3_times), int(tstop/dt)+ 1, len(step_amps)))

    for step_amp_idx, step_amp in enumerate(step_amps):
        for ramp3_time_idx, ramp3_time in enumerate(ramp3_times):
            for ramp3_amp_idx, ramp3_amp in enumerate(ramp3_amps):
                i_inj_mat[ramp3_amp_idx, ramp3_time_idx, :, step_amp_idx] = get_i_inj_double_ramp(ramp_amp, ramp3_amp,
                                                                                                  ramp3_time, step_amp,
                                                                                                  len_step,
                                                                                                  baseline_amp,
                                                                                                  len_ramp, start_ramp1,
                                                                                                  start_step,
                                                                                                  len_step2ramp,
                                                                                                  tstop, dt)
    return i_inj_mat


def get_inj_doubleramp_params(cell_id, run_idx, PP_params_dir='/home/cf/Phd/DAP-Project/cell_data/PP_params2.csv'):
    PP_params = pd.read_csv(PP_params_dir, header=0)
    PP_params['cell_id'].fillna(method='ffill', inplace=True)
    PP_params_cell = PP_params[PP_params['cell_id'] == cell_id].iloc[run_idx]
    dt = 0.01
    tstop = 691.99
    len_step = PP_params_cell['step_len']
    ramp_amp = PP_params_cell['ramp2_amp']
    ramp3_amps = [PP_params_cell['ramp3_amp']]
    baseline_amp = -0.05
    len_step2ramp = PP_params_cell['len_step2ramp']
    len_ramp = 2
    start_step = 222
    start_ramp1 = 20
    step_amps = [-0.1, 0.0, 0.1]
    ramp3_times = get_ramp3_times(PP_params_cell['delta_first'], PP_params_cell['delta_ramp'],
                                  PP_params_cell['len_ramp3_times'])
    return dict(ramp_amp=ramp_amp, ramp3_amps=ramp3_amps, ramp3_times=ramp3_times, step_amps=step_amps, len_step=len_step,
                baseline_amp=baseline_amp, len_ramp=len_ramp, start_ramp1=start_ramp1, start_step=start_step,
                len_step2ramp=len_step2ramp, tstop=tstop, dt=dt)
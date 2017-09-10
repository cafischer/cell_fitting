import numpy as np
from cell_fitting.new_optimization.fitter import iclamp_handling_onset
from cell_characteristics.analyze_APs import get_AP_onset_idxs


def get_slow_ramp(start_idx, end_idx, len_idx, step_amp):
    i_exp = np.zeros(len_idx)
    i_exp[start_idx:end_idx] = np.linspace(0, step_amp, end_idx - start_idx)
    return i_exp


def get_slow_ramp_reverse(start_idx, end_idx, len_idx, step_amp):
    i_exp = np.zeros(len_idx)
    i_exp[start_idx:end_idx] = np.linspace(1, step_amp, end_idx - start_idx)
    return i_exp


def get_step(start_idx, end_idx, len_idx, step_amp):
    i_exp = np.zeros(len_idx)
    i_exp[start_idx:end_idx] = step_amp
    return i_exp


def get_IV(cell, step_amp, step_fun, step_st_ms, step_end_ms, tstop, v_init=-75, dt=0.001):
    i_exp = step_fun(int(round(step_st_ms/dt)), int(round(step_end_ms/dt)), int(round(tstop/dt))+1, step_amp)

    # get simulation parameters
    simulation_params = {'sec': ('soma', None), 'i_inj': i_exp, 'v_init': v_init, 'tstop': tstop,
                         'dt': dt, 'celsius': 35, 'onset': 200}

    # record v
    v, t, _ = iclamp_handling_onset(cell, **simulation_params)

    return v, t, i_exp


def compute_time_to_1st_spike(v_traces, i_traces, t_trace):

    start_step = np.nonzero(i_traces[0])[0][0]
    amps = np.array([i_inj[start_step] for i_inj in i_traces])

    time_to_1st_spike = np.zeros(len(amps))
    for i, amp in enumerate(amps):
        AP_onsets = get_AP_onset_idxs(v_traces[i], threshold=0)
        if len(AP_onsets) == 0:
            time_to_1st_spike[i] = np.nan
        else:
            time_to_1st_spike[i] = t_trace[AP_onsets[0]] - t_trace[start_step]
    return amps, time_to_1st_spike
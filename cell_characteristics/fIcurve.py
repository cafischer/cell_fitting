import numpy as np
from cell_characteristics.analyze_APs import *


def compute_fIcurve(v_traces, i_traces, t_trace):
    start_step = np.nonzero(i_traces[0])[0][0]
    end_step = np.nonzero(i_traces[0])[0][-1] + 1
    dur_step = t_trace[end_step] - t_trace[start_step]

    amps = np.array([i_inj[start_step] for i_inj in i_traces])

    firing_rates = np.zeros(len(amps))
    for i, amp in enumerate(amps):
        AP_onsets = get_AP_onsets(v_traces[i], threshold=0)
        n_APs = len(AP_onsets)
        firing_rates[i] = n_APs / dur_step
    return amps[amps >= 0], firing_rates[amps >= 0]
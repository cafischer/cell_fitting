import numpy as np
from optimization.errfuns import rms
from statistics.ap_analyzer import ApAnalyzer

__author__ = 'caro'


# point-to-point
def errfun_pointtopoint(v_model, v_exp, *args):
    return rms(v_model, v_exp)


# feature-based
def errfun_featurebased(v_model, v_exp, t, *args):
    amp_model, width_model = get_AP_amp_width(v_model, t)
    amp_exp, width_exp = get_AP_amp_width(v_exp, t)

    if amp_model is None or width_model is None:
        return np.inf
    else:
        return rms(amp_model, amp_exp) + rms(width_model, width_exp)


def get_AP_amp_width(v, t):
    ap_analyzer = ApAnalyzer(v, t)
    dt = t[1] - t[0]
    vrest = np.mean(v[:100])
    AP_onsets = ap_analyzer.get_AP_onsets()
    if len(AP_onsets) == 0:
        return None, None
    else:
        AP_onset = AP_onsets[0]
        if len(AP_onsets) == 1:
            AP_end = -1
        else:
            AP_end = AP_onsets[1]
    AP_max = ap_analyzer.get_AP_max(AP_onset, AP_end, interval=5/dt)
    AP_amp = ap_analyzer.get_AP_amp(AP_max, vrest)
    AP_width = ap_analyzer.get_AP_width(AP_onset, AP_max, AP_end, vrest)
    return AP_amp, AP_width
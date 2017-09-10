from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import scipy.optimize
from cell_characteristics.analyze_APs import get_AP_max_idx, get_AP_amp, get_AP_width, get_DAP_amp, get_DAP_width, \
    get_fAHP_min_idx_using_splines, get_DAP_max_idx_using_splines, get_DAP_deflection, \
    get_AP_width_idxs, get_DAP_width_idx
from cell_fitting.util import init_nan


def get_spike_characteristics(AP_matrix, t_window, AP_interval, std_idxs, DAP_interval, v_rest,
                              order_fAHP_min, order_DAP_max, dist_to_DAP_max, check=False):
    dt = t_window[1] - t_window[0]
    AP_amp = init_nan(len(AP_matrix))
    AP_width_idxs = np.zeros((len(AP_matrix), 2), dtype=int)
    AP_width = init_nan(len(AP_matrix))
    fAHP_min_idx = init_nan(len(AP_matrix))
    DAP_max_idx = init_nan(len(AP_matrix))
    DAP_amp = init_nan(len(AP_matrix))
    DAP_deflection = init_nan(len(AP_matrix))
    DAP_width_idx = init_nan(len(AP_matrix))
    DAP_width = init_nan(len(AP_matrix))
    DAP_time = init_nan(len(AP_matrix))
    slope_start = init_nan(len(AP_matrix))
    slope_end = init_nan(len(AP_matrix))
    DAP_exp_slope = init_nan(len(AP_matrix))
    DAP_lin_slope = init_nan(len(AP_matrix))

    for i, AP_window in enumerate(AP_matrix):
        AP_max = get_AP_max_idx(AP_window, 0, len(AP_window), interval=AP_interval)
        if AP_max is None:
            continue
        AP_amp[i] = get_AP_amp(AP_window, AP_max, v_rest[i])
        AP_width_idxs[i, :] = get_AP_width_idxs(AP_window, t_window, 0, AP_max, AP_max + AP_interval, v_rest[i])
        AP_width[i] = get_AP_width(AP_window, t_window, 0, AP_max, AP_max + AP_interval, v_rest[i])

        std = np.std(AP_window[std_idxs[0]:std_idxs[1]])  # take first two ms for estimating the std
        w = np.ones(len(AP_window)) / std
        fAHP_min_idx[i] = get_fAHP_min_idx_using_splines(AP_window, t_window, AP_max, len(t_window),
                                                         order=order_fAHP_min, interval=AP_interval, w=w)
        if np.isnan(fAHP_min_idx[i]):
            continue

        DAP_max_idx[i] = get_DAP_max_idx_using_splines(AP_window, t_window, int(fAHP_min_idx[i]), len(t_window),
                                                       order=order_DAP_max,
                                                       interval=DAP_interval, dist_to_max=dist_to_DAP_max, w=w)
        if np.isnan(DAP_max_idx[i]):
            continue

        DAP_amp[i] = get_DAP_amp(AP_window, int(DAP_max_idx[i]), v_rest[i])
        DAP_deflection[i] = get_DAP_deflection(AP_window, int(fAHP_min_idx[i]), int(DAP_max_idx[i]))
        DAP_width_idx[i] = get_DAP_width_idx(AP_window, t_window, int(fAHP_min_idx[i]), int(DAP_max_idx[i]),
                                             len(t_window), v_rest[i])
        DAP_width[i] = get_DAP_width(AP_window, t_window, int(fAHP_min_idx[i]), int(DAP_max_idx[i]),
                                     len(t_window), v_rest[i])
        DAP_time[i] = t_window[int(round(DAP_max_idx[i]))] - t_window[AP_max]
        if np.isnan(DAP_width_idx[i]):
            continue

        half_fAHP_crossings = np.nonzero(np.diff(np.sign(AP_window[int(DAP_max_idx[i]):len(t_window)]
                                                         - AP_window[int(fAHP_min_idx[i])])) == -2)[0]
        if len(half_fAHP_crossings) == 0:
            continue

        half_fAHP_idx = half_fAHP_crossings[0] + DAP_max_idx[i]
        slope_start[i] = half_fAHP_idx  # int(round(DAP_width_idx[i] - 10/dt))
        slope_end[i] = len(t_window) - 1  # int(round(DAP_width_idx[i] + 20/dt))
        DAP_lin_slope[i] = np.abs((AP_window[int(slope_end[i])] - AP_window[int(slope_start[i])])
                                  / (t_window[int(slope_end[i])] - t_window[int(slope_start[i])]))

        def exp_fit(t, a):
            diff_exp = np.max(np.exp(-t / a)) - np.min(np.exp(-t / a))
            diff_points = AP_window[int(slope_start[i])] - AP_window[int(slope_end[i])]
            return (np.exp(-t / a) - np.min(np.exp(-t / a))) / diff_exp * diff_points + AP_window[int(slope_end[i])]

        DAP_exp_slope[i] = scipy.optimize.curve_fit(exp_fit,
                                                    np.arange(0, len(AP_window[int(slope_start[i]):int(slope_end[i])]),
                                                              1) * dt,
                                                    AP_window[int(slope_start[i]):int(slope_end[i])],
                                                    p0=1, bounds=(0, np.inf))[0]
    if check:
        check_measures(AP_matrix, t_window, AP_max, AP_width_idxs, AP_amp, AP_width, fAHP_min_idx,
                       DAP_max_idx, DAP_width_idx, DAP_amp, DAP_width, DAP_time,
                       slope_start, slope_end, DAP_exp_slope, DAP_lin_slope, v_rest)

    return AP_amp, AP_width, DAP_amp, DAP_deflection, DAP_width, DAP_time, DAP_lin_slope, DAP_exp_slope


def check_measures(AP_matrix, t_window, AP_max, AP_width_idxs, AP_amp, AP_width, fAHP_min_idx,
                   DAP_max_idx, DAP_width_idx, DAP_amp, DAP_width, DAP_time,
                   slope_start, slope_end, DAP_exp_slope, DAP_lin_slope, v_rest):
    dt = t_window[1] - t_window[0]
    for i, AP_window in enumerate(AP_matrix):
        print 'AP_amp (mV): ', AP_amp[i]
        print 'AP_width (ms): ', AP_width[i]
        if not np.isnan(DAP_max_idx[i]):
            print 'DAP_amp: (mV): ', DAP_amp[i]
            print 'DAP_width: (ms): ', DAP_width[i]
            print 'DAP time: (ms): ', DAP_time[i]
        if not np.isnan(DAP_exp_slope[i]):
            print 'DAP_exp_slope: ', DAP_exp_slope[i]
            print 'DAP_lin_slope: ', DAP_lin_slope[i]
        pl.figure()
        pl.plot(t_window, AP_window)
        pl.plot(t_window[AP_max], AP_window[AP_max], 'or', label='AP_max')
        pl.plot(t_window[AP_width_idxs[i, :]], AP_window[AP_width_idxs[i, :]], '-or', label='AP_width')
        if not np.isnan(fAHP_min_idx[i]):
            pl.plot(t_window[int(fAHP_min_idx[i])], AP_window[int(fAHP_min_idx[i])], 'og', label='fAHP')
            if not np.isnan(DAP_max_idx[i]):
                pl.plot(t_window[int(DAP_max_idx[i])], AP_window[int(DAP_max_idx[i])], 'ob', label='DAP_max')
            if not np.isnan(DAP_width_idx[i]):
                pl.plot([t_window[int(fAHP_min_idx[i])], t_window[int(DAP_width_idx[i])]],
                        [AP_window[int(fAHP_min_idx[i])] - (AP_window[int(fAHP_min_idx[i])] - v_rest[i]) / 2,
                         AP_window[int(DAP_width_idx[i])]],
                        '-ob', label='DAP_width')
            if not np.isnan(slope_start[i]) and not np.isnan(slope_end[i]):
                pl.plot([t_window[int(slope_start[i])], t_window[int(slope_end[i])]],
                        [AP_window[int(slope_start[i])], AP_window[int(slope_end[i])]],
                        '-oy', label='lin_slope')

                def exp_fit(t, a):
                    diff_exp = np.max(np.exp(-t / a)) - np.min(np.exp(-t / a))
                    diff_points = AP_window[int(slope_start[i])] - AP_window[int(slope_end[i])]
                    return (np.exp(-t / a) - np.min(np.exp(-t / a))) / diff_exp * diff_points + AP_window[
                        int(slope_end[i])]

                pl.plot(t_window[int(slope_start[i]): int(slope_end[i])],
                        exp_fit(np.arange(0, len(AP_window[int(slope_start[i]):int(slope_end[i])]), 1) * dt,
                                DAP_exp_slope[i]), 'y', label='exp_slope')
        pl.legend()
        pl.show()
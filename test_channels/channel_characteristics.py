from __future__ import division
import numpy as np
import matplotlib.pyplot as pl


def boltzmann_fun(v, vh, k):
    return 1 / (1+np.exp((v - vh)/k))


def steady_state_curve(v, vh, vs):
    return boltzmann_fun(v, -vh, vs)


def time_constant_curve(v, tau_min, tau_max, tau_delta, x_inf, vh, vs):
    return tau_min + (tau_max - tau_min) * x_inf * np.exp(tau_delta * (vh - v) / vs)


def std_steady_state_curve(v, vh, vs, vh_std, vs_std, kind='max'):
    curve1 = steady_state_curve(v, vh + vh_std, vs + vs_std)
    curve2 = steady_state_curve(v, vh + vh_std, vs - vs_std)
    curve3 = steady_state_curve(v, vh - vh_std, vs + vs_std)
    curve4 = steady_state_curve(v, vh - vh_std, vs - vs_std)

    curve = np.zeros(len(v))
    for i in range(len(v)):
        if kind == 'max':
            curve[i] = max(curve1[i], curve2[i], curve3[i], curve4[i])
        elif kind == 'min':
            curve[i] = min(curve1[i], curve2[i], curve3[i], curve4[i])
    return curve

if __name__ == '__main__':

    # fitted data


    # experimental data
    v_range = np.arange(-100, 0, 0.1)
    vh = -39
    k = 5
    vh_std = 5
    k_std = 0.9

    curve_act = boltzmann_fun(v_range, vh, k)
    curve_act_min = std_steady_state_curve(v_range, -vh, k, vh_std, k_std, kind='min')
    curve_act_max = std_steady_state_curve(v_range, -vh, k, vh_std, k_std, kind='max')

    pl.figure()
    pl.plot(v_range, curve_act, color='b')
    pl.fill_between(v_range, curve_act_max, curve_act_min, color='b', alpha=0.5)
    pl.show()

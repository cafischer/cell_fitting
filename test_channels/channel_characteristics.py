from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit


def boltzmann_fun(v, vh, k):
    return 1 / (1+np.exp((v - vh)/k))


def rate_constant(v, a, b, k):
    return (a * v + b) / (1 - np.exp((v + b / a) / k))


def tau(alpha, beta):
    return 1 / (alpha + beta)


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


def plot_activation_curves(v_range, vh_act, k_act, vh_std_act, k_std_act, vh_inact, k_inact, vh_std_inact, k_std_inact):
    curve_act = boltzmann_fun(v_range, vh_act, k_act)
    curve_act_min = std_steady_state_curve(v_range, -vh_act, k_act, vh_std_act, k_std_act, kind='min')
    curve_act_max = std_steady_state_curve(v_range, -vh_act, k_act, vh_std_act, k_std_act, kind='max')
    curve_inact = boltzmann_fun(v_range, vh_inact, k_inact)
    curve_inact_min = std_steady_state_curve(v_range, -vh_inact, k_inact, vh_std_inact, k_std_inact, kind='min')
    curve_inact_max = std_steady_state_curve(v_range, -vh_inact, k_inact, vh_std_inact, k_std_inact, kind='max')
    pl.figure()
    pl.plot(v_range, curve_act, color='r', label='activation')
    pl.fill_between(v_range, curve_act_max, curve_act_min, color='r', alpha=0.5)
    pl.plot(v_range, curve_inact, color='b', label='inactivation')
    pl.fill_between(v_range, curve_inact_max, curve_inact_min, color='b', alpha=0.5)
    pl.xlabel('V (mV)', fontsize=16)
    pl.ylabel('G (normalized)', fontsize=16)
    pl.legend(fontsize=16)
    pl.show()


def fit_time_constant(v_range, a_alpha, b_alpha, k_alpha, a_beta, b_beta, k_beta, vh, k, p0=(0.5, 10, 0.5)):

    alpha = rate_constant(v_range, a_alpha, b_alpha, k_alpha)
    beta = rate_constant(v_range, a_beta, b_beta, k_beta)
    time_constant = tau(alpha, beta)

    def curve_to_fit(v, tau_min, tau_max, tau_delta):
        m_inf = boltzmann_fun(v, vh, k)
        return time_constant_curve(v, tau_min, tau_max, tau_delta, m_inf, vh, -k)

    p_opt, p_cov = curve_fit(curve_to_fit, v_range, time_constant, p0=p0)
    time_constant_fit = curve_to_fit(v_range, *p_opt)
    return p_opt, time_constant_fit


if __name__ == '__main__':

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
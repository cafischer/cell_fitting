from __future__ import division
import pandas as pd
import matplotlib.pyplot as pl
from optimization.errfuns import rms
from test_channels.channel_characteristics import boltzmann_fun, time_constant_curve, rate_constant, compute_tau
import numpy as np
import scipy.optimize

# Note: Solution for ion channel does just hold if v is constant!!!

# load and transform data
i_traces = pd.read_csv('./plots/digitized_vsteps/traces.csv', index_col=0)
i_traces /= np.max(np.max(np.abs(i_traces)))

v_steps = np.arange(-60, -34, 5)
n_trials = 100
seed = 562
rand_gen = np.random.RandomState(seed)
bounds = [(-1, -0.0001), (-1, 1), (0.01, 10),
          (0.0001, 1), (-1, 1), (-10, -0.01)]


def compute_current(v, t, alpha_a_h, alpha_b_h, alpha_k_h, beta_a_h, beta_b_h, beta_k_h):
    vh_m = -44.4
    k_m = -5.2
    vh_h = -48.8
    k_h = 10
    h0 = 1
    p = 1
    q = 1
    e_ion = 60

    m_inf = boltzmann_fun(v, vh_m, k_m)
    h_inf = boltzmann_fun(v, vh_h, k_h)
    alpha_h = rate_constant(v, alpha_a_h, alpha_b_h, alpha_k_h) #/ 1000  # convert to /milliseconds
    beta_h = rate_constant(v, beta_a_h, beta_b_h, beta_k_h) #/ 1000  # convert to /milliseconds
    tau_h = compute_tau(alpha_h, beta_h)
    #h_inf = alpha_h / (alpha_h + beta_h)

    return m_inf ** p * (h_inf - (h_inf - h0) * np.exp(-t / tau_h)) ** q * (v-e_ion)


def fun_to_fit(candidate):
    t_vec = i_traces.index.values
    i_traces_fit = []
    for v_step in v_steps:
        i_traces_fit.append(compute_current(v_step, t_vec, *candidate))

    i_traces_fit = [i_trace_fit / np.max(np.abs(i_traces_fit)) for i_trace_fit in i_traces_fit]

    error = np.sum([rms(i_traces[str(v_step)], i_trace_fit)
              for v_step, i_trace_fit in zip(v_steps, i_traces_fit)])
    return error

# fitting
results = []
for trial in range(n_trials):
    x0 = np.array([rand_gen.uniform(*lh) for lh in bounds])
    #if trial == 0:
    #    x0 = [-2.88e-3, -4.9e-2, 4.63, 6.94e-3, 0.447, -2.63]
    res = scipy.optimize.minimize(fun_to_fit, x0=x0, method='L-BFGS-B', bounds=bounds)
    results.append(res)


errors = [res.fun for res in results]
best_fit = np.nanargmin(errors)
print results[best_fit]

# plot best result
pl.figure()
t_vec = i_traces.index.values
i_traces_fit = []
for v_step in v_steps:
    #x = [-2.88e-3, -4.9e-2, 4.63, 6.94e-3, 0.447, -2.63]
    #i_traces_fit.append(compute_current(v_step, t_vec, *x))
    i_traces_fit.append(compute_current(v_step, t_vec, *results[best_fit].x))

for v_step, i_trace_fit in zip(v_steps, i_traces_fit):
    pl.plot(t_vec, i_traces[str(v_step)], 'k')
    pl.plot(t_vec, i_trace_fit / np.max(np.abs(i_traces_fit)), 'r')
pl.show()


"""
#bounds = [(-5900, -5500), (0, 50), (0, 1000), (0, 1), (0, 1)]


def compute_current(v, t, fac, tau_min_h, tau_max_h, tau_delta_h, h0):
    vh_m = -44.4
    k_m = -5.2
    vh_h = -48.8
    k_h = 10
    p = 3
    q = 1
    m_inf = boltzmann_fun(v, vh_m, k_m)
    h_inf = boltzmann_fun(v, vh_h, k_h)
    tau_h = time_constant_curve(v, tau_min_h, tau_max_h, tau_delta_h, h_inf, vh_h, -k_h)
    return (
        fac
        * m_inf ** p
        * (h_inf - (h_inf - h0) * np.exp(-t / tau_h)) ** q
        * h_inf
    )
"""
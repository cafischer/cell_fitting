from __future__ import division
import pandas as pd
import matplotlib.pyplot as pl
from optimization.errfuns import rms
from test_channels.channel_characteristics import boltzmann_fun
import numpy as np
import scipy.optimize

# Note: Solution for ion channel does just hold if v is constant!!!

# load and transform data
i_traces = pd.read_csv('./plots/digitized_vsteps/traces.csv', index_col=0)
i_traces /= np.max(np.max(np.abs(i_traces)))

v_steps = np.arange(-60, -34, 5)
n_trials = 100
seed = 3
rand_gen = np.random.RandomState(seed)
bounds = [(0, 1), (0, 1), (0, 1000)]


def compute_current(v, t, m_inf, h_inf, tau_h):
    h0 = 1
    p = 1
    q = 1
    e_ion = 60
    return m_inf ** p * (h_inf - (h_inf - h0) * np.exp(-t / tau_h)) ** q * (v-e_ion)


def fun_to_fit(candidate):
    t_vec = i_traces.index.values
    i_traces_fit = []
    candidate = candidate.reshape(len(v_steps), len(bounds))
    for i, v_step in enumerate(v_steps):
        i_traces_fit.append(compute_current(v_step, t_vec, *candidate[i]))

    i_traces_fit = [i_trace_fit / np.max(np.abs(i_traces_fit)) for i_trace_fit in i_traces_fit]

    error = np.sum([rms(i_traces[str(v_step)], i_trace_fit)
              for v_step, i_trace_fit in zip(v_steps, i_traces_fit)])
    return error

# fitting
results = []
for trial in range(n_trials):
    x0 = np.ravel([[rand_gen.uniform(*lh) for lh in bounds] for i in range(len(v_steps))])
    bounds_all = sum([bounds for i in range(len(v_steps))], [])
    results.append(scipy.optimize.minimize(fun_to_fit, x0=x0, method='L-BFGS-B', bounds=bounds_all))


errors = [res.fun for res in results]
best_fit = np.nanargmin(errors)
print results[best_fit]

# plot best result
pl.figure()
t_vec = i_traces.index.values
i_traces_fit = []
candidate = results[best_fit].x
candidate = candidate.reshape(len(v_steps), len(bounds))
for i, v_step in enumerate(v_steps):
    i_traces_fit.append(compute_current(v_step, t_vec, *candidate[i]))

for v_step, i_trace_fit in zip(v_steps, i_traces_fit):
    pl.plot(t_vec, i_traces[str(v_step)], 'k')
    pl.plot(t_vec, i_trace_fit / np.max(np.abs(i_traces_fit)), 'r')
pl.show()

m_inf, h_inf, tau_h = results[best_fit].x.reshape(len(bounds), len(v_steps))

pl.figure()
pl.plot(v_steps, m_inf, 'r')
pl.plot(v_steps, h_inf, 'b')
pl.show()

pl.figure()
pl.plot(v_steps, tau_h)
pl.show()
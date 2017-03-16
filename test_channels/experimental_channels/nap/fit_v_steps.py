import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from test_channels.channel_characteristics import boltzmann_fun, time_constant_curve
from scipy.optimize import curve_fit

i_traces = pd.read_csv('./plots/digitized_vsteps/traces.csv', index_col=0)
v_range = np.arange(-60, -34, 5)
all_fits = list()
tau_m = list()
tau_h = list()
for v in v_range:
    t_range = i_traces.index.values
    v_time = np.ones(len(t_range)) * v
    v_time[i_traces[str(v)] == 0] = -80
    vh_m = -44.4
    k_m = -5.2
    vh_h = -48.8
    k_h = 10
    p = 4
    q = 1
    e_na = 60
    m0 = 0
    h0 = 1

    def fun_to_fit(x, g_max, tau_m, tau_h):
        v, t = zip(*x)
        v = np.array(v)
        t = np.array(t)
        m_inf = boltzmann_fun(v, vh_m, k_m)
        h_inf = boltzmann_fun(v, vh_h, k_h)
        return g_max * (m_inf - (m_inf - m0) * np.exp(-t / tau_m)) ** p \
               * (h_inf - (h_inf - h0) * np.exp(-t / tau_h)) ** q * (v - e_na)

    x_to_fit = zip(v_time, t_range)
    i_to_fit = i_traces[str(v)]
    bounds = [(0, 0, 0), (200, 500, 20000)]
    p_opt, p_cov = curve_fit(fun_to_fit, x_to_fit, i_to_fit, p0=(20, 50, 500),
                             bounds=bounds)
    #p_opt = [30, 0.00000000000000001, 100]
    print p_opt
    fit = fun_to_fit(x_to_fit, *p_opt)
    all_fits.append(fit)
    tau_m.append(p_opt[-2])
    tau_h.append(p_opt[-1])

pl.figure()
for v, fit in zip(v_range, all_fits):
    pl.plot(t_range, i_traces[str(v)], 'k')
    pl.plot(t_range, fit, 'r')
pl.show()

# -35: [  13.22874343    1.            0.79226462  234.40276684   40.09653709]
# -40: [  12.40475889    0.93718608    0.73924065  130.38481352   29.80951733]
# -45: [ 19.75975088   0.68272233   0.71122601  91.91255087  20.41331681]
# -50: [  3.90384795e+01   3.18420661e-01   1.00000000e+00   5.10459225e+01  3.25520404e+02]
# -55: [  2.15790403e+02   1.25882964e-01   1.00000000e+00   5.24758163e+01  3.00968635e+02]
# -60 [  1.00000000e+02   1.46673964e-01   4.47163694e-01   5.00000000e+02  1.94765083e+02]

# plot tau
pl.figure()
pl.plot(v_range, tau_m, 'b')
pl.plot(v_range, tau_h, 'g')
pl.show()

# fit taus
def curve_to_fit(v, tau_min, tau_max, tau_delta):
    m_inf = boltzmann_fun(v, vh_m, k_m)
    return time_constant_curve(v, tau_min, tau_max, tau_delta, m_inf, vh_m, -k_m)

p_opt, p_cov = curve_fit(curve_to_fit, v_range, tau_m, p0=(1, 100, 0.2)) #, bounds=[(0, 0, 0), (100, 1000, 1.5)])

for n, p in zip(['m_tau_min', 'm_tau_max', 'm_tau_delta'], p_opt):
    print n+' = ', p
time_constant_fit = curve_to_fit(v_range, *p_opt)

pl.figure()
pl.plot(v_range, tau_m, 'ok', label='exp')
pl.plot(v_range, time_constant_fit, 'r', label='fit')
pl.xlabel('V (mV)')
pl.ylabel('Tau (ms)')
pl.legend()
pl.show()


def curve_to_fit(v, tau_min, tau_max, tau_delta):
    m_inf = boltzmann_fun(v, vh_h, k_h)
    return time_constant_curve(v, tau_min, tau_max, tau_delta, m_inf, vh_h, -k_h)

p_opt, p_cov = curve_fit(curve_to_fit, v_range, tau_h, p0=(0.1, 1, 0.2)) #, bounds=[(0, 0, 0), (1000, 15000, 1.5)])

for n, p in zip(['h_tau_min', 'h_tau_max', 'h_tau_delta'], p_opt):
    print n+' = ', p
time_constant_fit = curve_to_fit(v_range, *p_opt)

pl.figure()
pl.plot(v_range, tau_h, 'ok', label='exp')
pl.plot(v_range, time_constant_fit, 'r', label='fit')
pl.xlabel('V (mV)')
pl.ylabel('Tau (ms)')
pl.legend()
pl.show()


# next steps: simulate channel with fitted parameters -> check reproduce v steps -> play around with fits


"""
# testing
i_traces = pd.read_csv('./plots/digitized_vsteps/traces.csv', index_col=0)
v_steps = np.arange(-60, -35, 5)
y_offsets = [0, 0, 0, 0, -10, -30]

for v, y_offset in zip(v_steps, y_offsets):
    t_range = i_traces.index.values

    not_nan = np.logical_not(np.isnan(i_traces[str(v)])).values
    t_to_fit = t_range[not_nan]
    i_to_fit = i_traces[str(v)][not_nan]
    min_i = np.min(i_to_fit)
    max_i = np.max(i_to_fit)
    def fun_to_fit(t, tau_m):
        return (max_i-min_i) * (1 - np.exp(-(t+y_offset) / tau_m)) + min_i

    p_opt, p_cov = curve_fit(fun_to_fit, t_to_fit, i_to_fit, p0=(10))
    print 'tau = ' + str(p_opt)

    pl.figure()
    pl.plot(t_to_fit, i_to_fit, 'k')
    pl.plot(t_range, fun_to_fit(t_range, *p_opt), 'r')
    pl.ylim(-800, 0)
    pl.show()
"""
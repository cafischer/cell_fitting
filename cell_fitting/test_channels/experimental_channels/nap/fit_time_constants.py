from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
from test_channels.channel_characteristics import fit_time_constant, boltzmann_fun, time_constant_curve, \
    rate_constant, compute_tau

v_range = np.arange(-200, 100, 0.01)

# nap channel (magistretti)

# fit time constant
vh = -48.8
k = 10
alpha = rate_constant(v_range, -2.88 * 1e-3, -4.9 * 1e-2, 4.63)
beta = rate_constant(v_range, 6.94 * 1e-3, 0.447, -2.63)
time_constant = compute_tau(alpha, beta)
p_opt, time_constant_fit = fit_time_constant(v_range,
                                             -2.88 * 1e-3, -4.9 * 1e-2, 4.63,
                                             6.94 * 1e-3, 0.447, -2.63,
                                             vh, k, p0=(0.1, 10, 0.5))

def curve_to_fit(v, tau_min, tau_max, tau_delta):
    m_inf = boltzmann_fun(v, vh, k)
    return time_constant_curve(v, tau_min, tau_max, tau_delta, m_inf, vh, -k)
p_opt[:2] *= 1000
for n, p in zip(['tau_min', 'tau_max', 'tau_delta'], p_opt):
    print n+' = ', p
time_constant_fit = curve_to_fit(v_range, *p_opt)

pl.figure()
pl.plot(v_range, time_constant * 1000, color='blue', label='exp')
pl.plot(v_range, time_constant_fit, color='red', label='fit')
pl.xlabel('V (mV)')
pl.ylabel('Tau (ms)')
pl.xlim([-100, 100])
pl.legend()
pl.show()


# fit with temperature adjustment
q10 = 3
degree_now = 35
degree_channel = 20
t_adj = q10**((degree_now - degree_channel) / 10)

p_opt[:2] *= 1/t_adj
print 'with temperature adjustment: '
for n, p in zip(['tau_min', 'tau_max', 'tau_delta'], p_opt):
    print n+' = ', p
time_constant_fit = curve_to_fit(v_range, *p_opt)

pl.figure()
pl.plot(v_range, time_constant * 1000 * 1/t_adj, color='blue', label='exp')
pl.plot(v_range, time_constant_fit, color='red', label='fit')
pl.xlabel('V (mV)')
pl.ylabel('Tau (ms)')
pl.xlim([-100, 100])
pl.legend()
pl.show()
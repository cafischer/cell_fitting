import matplotlib.pyplot as pl
import numpy as np
from test_channels.channel_characteristics import fit_time_constant
from test_channels.experimental_channels.hcn.hcn_channel import rate_constant, tau


v_range = np.arange(-150, 0, 0.001)

# hcn channel (dickson)

# fit fast time constant
alpha = rate_constant(v_range, -2.89 * 1e-3, -0.445, 24.02)
beta = rate_constant(v_range, 2.71 * 1e-2, -1.024, -17.4)
time_constant_fast = tau(alpha, beta)
p_opt, time_constant_fast_fit = fit_time_constant(v_range,
                                                  -2.89 * 1e-3, -0.445, 24.02,
                                                  2.71 * 1e-2, -1.024, -17.4,
                                                  -67.4, 12.66, p0=(0.5, 10, 0.5))

pl.figure()
pl.plot(v_range, time_constant_fast, color='blue', label='fast')
pl.plot(v_range, time_constant_fast_fit, color='red', label='fit')
pl.xlabel('V (mV)')
pl.ylabel('Tau (ms)')
pl.legend()
#pl.show()

# fit slow time constant
alpha = rate_constant(v_range, -3.18 * 1e-3, -0.695, 26.72)
beta = rate_constant(v_range, 2.16 * 1e-2, -1.065, -14.25)
time_constant_slow = tau(alpha, beta)
p_opt, time_constant_slow_fit = fit_time_constant(v_range,
                                                  -3.18 * 1e-3, -0.695, 26.72,
                                                  2.16 * 1e-2, -1.065, -14.25,
                                                  -57.92, 9.26, p0=(0.5, 10, 0.5))

pl.figure()
pl.plot(v_range, time_constant_slow, color='blue', label='slow')
pl.plot(v_range, time_constant_slow_fit, color='red', label='fit')
pl.xlabel('V (mV)')
pl.ylabel('Tau (ms)')
pl.legend()
pl.show()

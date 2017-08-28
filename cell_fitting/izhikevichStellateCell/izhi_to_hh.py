from izhikevichStellateCell import *
import scipy.optimize
import numpy as np


def sigmoid(v, v_half, k):
    return 1/(1+np.exp((v_half-v)/k))

def tau(v, c_base, c_amp, v_max, sig):
    return c_base + c_amp * np.exp(-(v_max - v)**2 / sig**2)

# define HH model
gl = 0.1
gna = 1
gh = 1
El = -70
Ena = 60
Eh = -21
cm = 200
i_inj = 0

def m_inf(v):
    return sigmoid(v, -50, 4)

def h_inf(v):
    return sigmoid(v, -58, -9)

def tau_h(v):
    return tau(v, 100, 300, -65, 30)


def v_cline(v, i_inj, gl, gna, gh, El, Ena, Eh):
    return (i_inj - gl * (v - El) - gna * m_inf(v) * (v - Ena)) / (gh * (v - Eh))

def h_cline(v):
    return h_inf(v)

v_range = np.arange(-80, -30, 0.01)
pl.figure()
pl.plot(v_range, v_cline(v_range, i_inj, gl, gna, gh, El, Ena, Eh))
pl.plot(v_range, h_cline(v_range))
pl.gca().invert_yaxis()
#pl.ylim(0.4, 0)
pl.show()



# define Izhikevich model
cm = 185
k_rest = 0.75
k_t = 200
a1 = 0.0072
b1 = 28.21
d1 = 0.73
a2 = 1.026
b2 = 2.049
d2 = -531.63
v_rest = -62.5
v_t = -47.0
v_reset = -49.0
v_peak = 51.5
i_b = 0
v0 = v_rest
u0 = [0, 0]

# plot phase plane
vmin = -80
vmax = 0
umin = -100
umax = 200
#phase_plot(vmin, vmax, umin, umax, v_rest, v_t, cm, k_rest, a1, b1, i_b)

# approximate null clines for Hodgkin-Huxley model
i_inj = 0
v_range = np.arange(-70, -50, 0.1)


# u-nullcline
u_cline_data = a1 * (b1 * (v_range - v_rest))
u_cline_normalized = (u_cline_data - umin) / (umax - umin)
p_opt, p_cov = scipy.optimize.curve_fit(sigmoid, v_range, u_cline_normalized, p0=(-60, 1))
print p_opt

pl.figure()
pl.plot(v_range, u_cline_normalized, 'k')
pl.plot(v_range, sigmoid(v_range, *p_opt), 'r')
pl.show()

# v-nullcline
m_half = -50
m_k = 4
def m_inf(v):
    return sigmoid(v, m_half, m_k)
El = -70
Eh = -21
Ena = 60

def v_cline(v, gl, gna, gh):
    return (i_inj - gl * (v - El) - gna * m_inf(v) * (v - Ena)) / (gh * (v - Eh))

v_cline_data = k_rest * (v_range - v_rest) * (v_range - v_t) + i_inj
v_cline_normalized = (v_cline_data - umin) / (umax - umin)

p_opt, p_cov = scipy.optimize.curve_fit(v_cline, v_range, v_cline_normalized)
print p_opt

pl.figure()
pl.plot(v_range, v_cline_normalized, 'k')
pl.plot(v_range, v_cline(v_range, *p_opt), 'r')
pl.show()
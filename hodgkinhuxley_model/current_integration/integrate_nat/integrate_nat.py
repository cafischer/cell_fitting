from __future__ import division
import numpy as np
import pylab as pl
import pandas as pd
import scipy.optimize
from scipy.integrate import ode


# functions
def rates(vm, Ra, Rb, qa, tha, Rd, Rg, qi, thi1, thi2, thinf, qinf):
    a = Ra * qa * efun((tha - vm)/qa)
    b = Rb * qa * efun((vm - tha)/qa)

    mtau = 1/(a+b)
    minf = a/(a+b)

    a = Rd * qi * efun((thi1 - vm)/qi)
    b = Rg * qi * efun((vm - thi2)/qi)

    htau = 1/(a+b)
    hinf = 1/(1+np.exp((vm-thinf)/qinf))

    return mtau, minf, htau, hinf


def efun(x):
    if np.abs(x) < 1e-6:
        y = 1 - x/2
    else:
        y = x/(np.exp(x) - 1)
    return y

# parameters from simulation
ena = 80

# channel parameter
gbar = 10  # (pS/um2)
tha = -35  # (mV)
qa = 9  # (mV)
Ra = 0.182  # (/ms)
Rb = 0.124  # (/ms)
thi1 = -50  # (mV)
thi2 = -75  # (mV)
qi = 5  # (mV)
thinf = -65  # (mV)
qinf = 6.2  # (mV)
Rg = 0.0091  # (/ms)
Rd = 0.024  # (/ms)

# load data
data = pd.read_csv('./nat_neuronsim.csv')
v = np.array(data.v)
t = np.array(data.t)
dt = t[1] - t[0]

# initialization
mtau, minf, htau, hinf = rates(v[0], Ra, Rb, qa, tha, Rd, Rg, qi, thi1, thi2, thinf, qinf)
m0 = minf
h0 = hinf
ina0 = 1e-4 * gbar*m0**3*h0 * (v[0] - ena)

# integration: vode
def dmhdt(ts, y):
    [m, h] = y
    vs = np.interp(ts, t, v)  # interpolate V
    mtau, minf, htau, hinf = rates(vs, Ra, Rb, qa, tha, Rd, Rg, qi, thi1, thi2, thinf, qinf)

    return [(minf-m)/mtau, (hinf-h)/htau]

solve_mh = ode(dmhdt).set_integrator('vode', method='bdf')
solve_mh.set_initial_value([m0, h0], 0)

ina_vode = np.zeros(len(t))
ina_vode[0] = ina0

idx = 0
while solve_mh.successful() and idx < len(t)-1:
    idx += 1
    solve_mh.integrate(solve_mh.t+dt)
    [m, h] = solve_mh.y

    vs = np.interp(solve_mh.t, t, v)
    ina_vode[idx] = 1e-4 * gbar*m**3*h * (vs - ena)


# integrate assuming minf, hinf, mtau, htau constant
ina_ana = np.zeros(len(t))
ina_ana[0] = ina0
idx = 0
ts = 0
m = m0
h = h0
while idx < len(t)-1:
    idx += 1
    ts += dt

    vs = np.interp(ts, t, v)
    mtau, minf, htau, hinf = rates(vs, Ra, Rb, qa, tha, Rd, Rg, qi, thi1, thi2, thinf, qinf)
    m = minf + (m - minf) * np.exp(-dt / mtau)
    h = hinf + (h - hinf) * np.exp(-dt / htau)

    ina_ana[idx] = 1e-4 * gbar*m**3*h * (vs - ena)

# integration: Implicit (backward) Euler

def fun_m(m, m_old, dt, minf, mtau):
    return m - m_old - dt * ((minf-m)/mtau)

def fun_h(h, h_old, dt, hinf, htau):
    return h - h_old - dt * ((hinf-h)/htau)

m_impeuler = np.zeros(len(t))
m_impeuler[0] = m0
h_impeuler = np.zeros(len(t))
h_impeuler[0] = h0
ina_impeuler = np.zeros(len(t))
ina_impeuler[0] = ina0

for i in range(1, len(t)):
    vs = v[i]
    mtau, minf, htau, hinf = rates(vs, Ra, Rb, qa, tha, Rd, Rg, qi, thi1, thi2, thinf, qinf)

    m_impeuler[i] = scipy.optimize.newton(fun_m, 0, args=(m_impeuler[i-1], dt, minf, mtau))
    h_impeuler[i] = scipy.optimize.newton(fun_h, 0, args=(h_impeuler[i-1], dt, hinf, htau))

    ina_impeuler[i] = 1e-4 * gbar*m_impeuler[i]**3*h_impeuler[i] * (v[i-1] - ena)


# integration: Exponential Euler

def phi(z):
    return (np.exp(z) - 1) / z

def slope_intercept_mh(vm):
    mtau, minf, htau, hinf = rates(vm, Ra, Rb, qa, tha, Rd, Rg, qi, thi1, thi2, thinf, qinf)
    return -1/mtau, -1/htau, minf/mtau, hinf/htau

m_slope = np.zeros(len(t))
h_slope = np.zeros(len(t))
m_intercept = np.zeros(len(t))
h_intercept = np.zeros(len(t))
m_slope[0], h_slope[0], m_intercept[0], h_intercept[0] = slope_intercept_mh(v[0])
for i in range(1, len(t)):
    m_slope[i], h_slope[i], m_intercept[i], h_intercept[i] = slope_intercept_mh(v[i-1])  # v from the previous time step

m_expeuler = np.zeros(len(t))
m_expeuler[0] = m0
h_expeuler = np.zeros(len(t))
h_expeuler[0] = h0
ina_expeuler = np.zeros(len(t))
ina_expeuler[0] = ina0

for i in range(1, len(t)):
    m_expeuler[i] = dt * phi(m_slope[i]*dt) * m_intercept[i-1] + m_expeuler[i-1] * np.exp(m_slope[i]*dt)
    h_expeuler[i] = dt * phi(h_slope[i]*dt) * h_intercept[i-1] + h_expeuler[i-1] * np.exp(h_slope[i]*dt)

    # compute current
    ina_expeuler[i] = 1e-4 * gbar*m_expeuler[i]**3*h_expeuler[i] * (v[i-1] - ena)  # v from the previous time step


with open('nat_neuronsim.npy', 'r') as f:
    ina_neuron = np.load(f)

pl.figure()
pl.plot(t, ina_neuron, 'r', linewidth=1.5, label='NEURON \nsimulation')
pl.plot(t, ina_vode, 'g', linewidth=1.5, label='Vode (bdf)')
pl.plot(t, ina_ana, 'b', linewidth=1.5, label='Semi-analytic')
pl.plot(t, ina_impeuler, 'k', linewidth=1.5, label='Implicit Euler')
pl.plot(t, ina_expeuler, 'y', linewidth=1.5, label='Exponential Euler')
pl.xlabel('$Time (ms)$', fontsize=18)
pl.ylabel('$Current (pS/\mu m^2)$', fontsize=18)
pl.legend(loc='lower right', fontsize=18)
pl.show()
from __future__ import division
import numpy as np
import pylab as pl
import pandas as pd
from scipy.integrate import ode


# functions
def rates(v, q10_act, q10_inact, celsius, temp, zetan, pw, tq, qq, vhalfn, gmn, a0n, nmin, nscale,
          zetal, vhalfl, lmin, lscale):

    tadj_act = q10_act**((celsius-temp)/10)
    tadj_inact = q10_inact**((celsius-temp)/10)

    a = alpn(v, zetan, pw, tq, qq, vhalfn, celsius)
    ninf = 1/(1 + a)
    taun = (betn(v, pw, tq, qq, gmn, vhalfn, celsius)/(a0n*(1+a))) / tadj_act
    if taun<nmin:
        taun=nmin
    taun=taun/nscale

    a = alpl(v, zetal, vhalfl, celsius)
    linf = 1/(1 + a)
    taul = (0.26*(v+50)) / tadj_inact
    if taul<lmin:
        taul=lmin
    taul=taul/lscale

    return taun, ninf, taul, linf


def alpn(v, zetan, pw, tq, qq, vhalfn, celsius):
    zeta=zetan+pw/(1+np.exp((v-tq)/qq))
    return np.exp(zeta*(v-vhalfn)*1.e-3*9.648e4/(8.315*(273.16+celsius)))


def betn(v, pw, tq, qq, gmn, vhalfn, celsius):
    zeta=zetan+pw/(1+np.exp((v-tq)/qq))
    return np.exp(zeta*gmn*(v-vhalfn)*1.e-3*9.648e4/(8.315*(273.16+celsius)))


def alpl(v, zetal, vhalfl, celsius):
    return np.exp(zetal*(v-vhalfl)*1.e-3*9.648e4/(8.315*(273.16+celsius)))


# derivatives
def dnl(t, y, q10_act, q10_inact, celsius, temp, zetan, pw, tq, qq, vhalfn, gmn, a0n, nmin, nscale,
          zetal, vhalfl, lmin, lscale):
    [n, l] = y
    taun, ninf, taul, linf = rates(v[t], q10_act, q10_inact, celsius, temp, zetan, pw, tq, qq, vhalfn, gmn, a0n, nmin, nscale,
          zetal, vhalfl, lmin, lscale)
    return [(ninf-n)/taun, (linf-l)/taul]

# parameters from simulation
celsius = 36
ek = -87

# load data
data = pd.read_csv('../../data/cell_2013_12_13f/dap/dap_reproduced.csv').convert_objects(convert_numeric=True)
v = np.array(data.v)
t = np.array(data.t)
dt = t[1] - t[0]
tstop = t[-1]
v_init = v[0]
t_idx = np.arange(0, len(t))
dt_idx = 1

# channel parameter
gbar = 1  # (S/cm2)
vhalfn = 11  # (mV)
a0n = 0.05  # (/ms)
zetan = -1.5  # (1)
gmn = 0.55  # (1)
pw = -1  # (1)
tq = -40  # (mV)
qq = 5  # (mV)
nmin = 0.1  # (ms)
nscale = 1
vhalfl = -56  # (mV)
a0l = 0.05  # (/ms)
zetal = 3  # (1)
lmin = 2  # (ms)
lscale = 1
q10_act = 5
q10_inact = 1
temp = 24  # (degC)

# initialization
taun, ninf, taul, linf = rates(v[0], q10_act, q10_inact, celsius, temp, zetan, pw, tq, qq, vhalfn, gmn, a0n, nmin, nscale,
          zetal, vhalfl, lmin, lscale)
n = ninf
l = linf
ik = np.zeros(len(t))

# create ode solver
solve_nl = ode(dnl).set_integrator('vode', method='bdf')
solve_nl.set_initial_value([n, l], 0)
solve_nl.set_f_params(q10_act, q10_inact, celsius, temp, zetan, pw, tq, qq, vhalfn, gmn, a0n, nmin, nscale,
          zetal, vhalfl, lmin, lscale)

# integration
while solve_nl.successful() and solve_nl.t < t_idx[-1]:
    solve_nl.integrate(solve_nl.t+dt_idx)
    [m, h] = solve_nl.y

    ik[solve_nl.t] = gbar*n*l * (v[solve_nl.t] - ek)


pl.figure()
pl.plot(t, ik)
pl.show()
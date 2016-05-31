from __future__ import division
import numpy as np
import pylab as pl
import pandas as pd
from scipy.integrate import ode


# functions
def rates(v, Ra, Rb, qa, tha, celsius, temp):
    a = Ra * qa * efun((tha - v)/qa)
    b = Rb * qa * efun((v - tha)/qa)

    tadj = q10**((celsius - temp)/10)

    mtau = 1/tadj/(a+b)
    minf = a/(a+b)

    a = Rd * qi * efun((thi1 - v)/qi)
    b = Rg * qi * efun((v - thi2)/qi)

    htau = 1/tadj/(a+b)
    hinf = 1/(1+np.exp((v-thinf)/qinf))

    return mtau, minf, htau, hinf


def efun(x):
    if np.abs(x) < 1e-6:
        y = 1 - x/2
    else:
        y = x/(np.exp(x) - 1)
    return y


# slope, offset
def slopes_mh(i):
    mtau, minf, htau, hinf = rates(v[i]+vshift, Ra, Rb, qa, tha, celsius, temp)
    return [minf/mtau, hinf/htau]


def offsets_mh(i):
    mtau, minf, htau, hinf = rates(v[i]+vshift, Ra, Rb, qa, tha, celsius, temp)
    return [-1/mtau, -1/htau]


# derivatives
def dmh(t, y, vshift, Ra, Rb, qa, tha, celsius, temp):
    [m, h] = y

    mtau, minf, htau, hinf = rates(v[t]+vshift, Ra, Rb, qa, tha, celsius, temp)

    mslope = -1/mtau
    moffset = minf/mtau
    hslope = -1/htau
    hoffset = hinf/htau

    return [mslope*m+moffset, hslope*h+hoffset]


# parameters from simulation
celsius = 36
ena = 83

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
vshift = 0  # (mV)
gbar = 1  # (pS/um2)
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
temp = 23  # (degC)		
q10 = 3
vin = -120  # (mV)
vax = 100  # (mV)
 
# initialization
mtau, minf, htau, hinf = rates(v[0]+vshift, Ra, Rb, qa, tha, celsius, temp)
m = minf
h = hinf
ina = np.zeros(len(t))

# create ode solver
solve_mh = ode(dmh).set_integrator('vode', method='bdf')
#solve_mh = ode(dmh).set_integrator('dopri5')
solve_mh.set_initial_value([m, h], 0)
solve_mh.set_f_params(vshift, Ra, Rb, qa, tha, celsius, temp)

# integration
while solve_mh.successful() and solve_mh.t < t_idx[-1]:
    solve_mh.integrate(solve_mh.t+dt_idx)
    [m, h] = solve_mh.y

    ina[solve_mh.t] = 1e-4 * gbar*m*m*m*h * (v[solve_mh.t] - ena)


pl.figure()
pl.plot(t, ina)
pl.show()
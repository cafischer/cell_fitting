from __future__ import division
import numpy as np
import pylab as pl


def phi(z):
    return (np.exp(z) - 1) / z

# dv/dt = a(t) * v(t) + b(t)

t = np.arange(0, 5, 0.01)
dt = t[1] - t[0]
a = 2 * t
b = t

v = np.zeros(len(t))
v_analytic = 0.5 * np.exp(t**2) - 0.5

# integration
for i in range(1, len(t)):
    v[i] = dt * phi(a[i]*dt) * b[i-1] + v[i-1] * np.exp(a[i]*dt)

pl.figure()
pl.plot(t, v_analytic, 'k', linewidth=2)
pl.plot(t, v, 'r')
pl.show()
from __future__ import division
import numpy as np
import pandas as pd
from ode_solver import solve_implicit_euler
import matplotlib.pyplot as pl
from neuron import h
from hodgkinhuxley_model.mechanisms import IonChannel
from hodgkinhuxley_model.cell import Cell
from hodgkinhuxley_model.hh_solver import HHSolver

__author__ = 'caro'


def gradient(theta, v_star, t, v, dvdtheta):

    # compute error for each parameter (theta)
    derrordtheta = np.zeros(len(theta))
    error = np.zeros(len(theta))

    for j in range(len(theta)):

        derrordtheta[j] = 1.0/len(t) * np.sum((v - v_star) * dvdtheta[j])

        error[j] = 1.0/len(t) * np.sum(0.5 * (v - v_star)**2)

    return derrordtheta, error


if __name__ == '__main__':

    # make model

    # make naf ionchannel
    g_max = 0.5
    ep = 80
    n_gates = 2
    power_gates = [3, 1]

    vshift = -3.5
    gbar = 0.0  # (S/cm2)

    mv = 38
    ms = 10
    mvtau = 30
    mctau = 0.14
    motau = 0.025

    hv = 62.9
    hs = 8
    hvtau = 37
    hstau = 15
    hctau = 1.15
    hotau = 0.15

    def inf_gates(v):
        minf = 1/(1 + np.exp((-(v + vshift) - mv) / ms))
        hinf = 1/(1 + np.exp(((v + vshift) + hv) / hs))
        return np.array([minf, hinf])

    def tau_gates(v):
        if(v + vshift) < -30.0:
            mtau = motau + mctau * np.exp(((v + vshift) + mvtau) / ms)
        else:
            mtau = motau + mctau * np.exp((-(v + vshift) - mvtau) / ms)
        htau = hotau + hctau / (1 + np.exp(((v + vshift) + hvtau) / hstau))
        return np.array([mtau, htau])

    naf = IonChannel(g_max, ep, n_gates, power_gates, inf_gates, tau_gates)

    """
    # make ka ionchannel
    g_max = 0.07
    ep = -80
    n_gates = 2
    power_gates = [1, 1]

    celsius = 35
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

    def alpn(v, zetan, pw, tq, qq, vhalfn, celsius):
        zeta=zetan+pw/(1+np.exp((v-tq)/qq))
        return np.exp(zeta*(v-vhalfn)*1.e-3*9.648e4/(8.315*(273.16+celsius)))

    def betn(v, zetan, pw, tq, qq, gmn, vhalfn, celsius):
        zeta=zetan+pw/(1+np.exp((v-tq)/qq))
        return np.exp(zeta*gmn*(v-vhalfn)*1.e-3*9.648e4/(8.315*(273.16+celsius)))

    def alpl(v, zetal, vhalfl, celsius):
        return np.exp(zetal*(v-vhalfl)*1.e-3*9.648e4/(8.315*(273.16+celsius)))

    def inf_gates(v):
        a = alpn(v, zetan, pw, tq, qq, vhalfn, celsius)
        ninf = 1/(1 + a)
        a = alpl(v, zetal, vhalfl, celsius)
        linf = 1/(1 + a)
        return np.array([ninf, linf])

    def tau_gates(v):
        tadj_act = q10_act**((celsius-temp)/10)
        tadj_inact = q10_inact**((celsius-temp)/10)

        a = alpn(v, zetan, pw, tq, qq, vhalfn, celsius)
        b = betn(v, zetan, pw, tq, qq, gmn, vhalfn, celsius)
        taun = (b/(a0n*(1+a))) / tadj_act
        if taun<nmin:
            taun=nmin
        taun=taun/nscale

        taul = (0.26*(v+50)) / tadj_inact
        if taul<lmin:
            taul=lmin
        taul=taul/lscale
        return np.array([taun, taul])

    ka = IonChannel(g_max, ep, n_gates, power_gates, inf_gates, tau_gates)
    """

    # create cell
    cm = 1
    length = 16
    diam = 8
    ionchannels = [naf]

    from fit_currents.error_analysis.model_generator import from_protocol
    def i_inj(ts):
        times, amps, amp_types = from_protocol('ramp')
        if len(np.nonzero(ts <= np.array(times))[0]) == 0:
            idx = len(times)-1  # in case it asks for later times return as if for last section
        else:
            idx = np.nonzero(ts <= np.array(times))[0][0]
        if idx > 0:
            idx -= 1
        if amp_types[idx] == 'const':
            return amps[idx]
        elif amp_types[idx] == 'rampup' or amp_types[idx] == 'rampdown':
            return (amps[idx+1]-amps[idx]) / (times[idx+1] - times[idx]) * (ts-times[idx]) + amps[idx]
        else:
            return None

    cell = Cell(cm, length, diam, ionchannels, i_inj)

    # create odesolver
    data_dir = './testdata/modeldata.csv'
    data = pd.read_csv(data_dir)

    v_star = np.array(data.v)
    t = np.array(data.t)
    v0 = v_star[0]
    i_inj = np.array(data.i)
    y0 = 0
    hhsolver = HHSolver('ImplicitEuler')
    dtheta = 0.001
    theta_max = 1
    theta_range = np.arange(0, theta_max+dtheta, dtheta)

    v = np.zeros(len(theta_range), dtype=object)
    y = np.zeros(len(theta_range), dtype=object)

    for i, theta in enumerate(theta_range):
        cell.ionchannels[0].g_max = theta
        v[i], _, _, y[i] = hhsolver.solve_adaptive_y(cell, t, v_star, v0, y0, 0)
        #v[i], _, _ = hhsolver.solve_adaptive(cell, t, v0)
        #v[i], current, p_gates = hhsolver.solve(cell, t, v0, np.array(data.i))

        #pl.figure()
        #pl.plot(t, v_star, 'k')
        #pl.plot(t, v[i], 'b')
        #pl.show()

    # numerical dvdtheta
    dvdtheta_quotient = np.zeros((len(theta_range), len(t)))
    for ts in range(len(t)):
        v_ts = np.array([v[i][ts] for i in range(len(theta_range))])
        dvdtheta_quotient[:, ts] = np.gradient(v_ts, dtheta)

    # compare y and numerical dvdtheta
    #for i, theta in enumerate(theta_range):
    #    pl.figure()
    #    pl.plot(t, dvdtheta_quotient[i, :], 'k', label='num. dvdtheta')
    #    pl.plot(t, y[i], 'r', label='y')
    #    pl.legend()
    #    pl.show()

    # compare numerical derrordtheta with derrordtheta with Euler dvdtheta
    derrordtheta_euler = np.zeros((2, len(theta_range)))
    error = np.zeros((2, len(theta_range)))
    for i, theta in enumerate(theta_range):
        g = [theta, 0.07]
        derrordtheta_euler[:, i], error[:, i] = gradient(g, v_star, t, v[i], [y[i], 0])

    derrordtheta_quotient = np.gradient(error[0, :], dtheta)  # np.diff(error[0, :]) / dtheta

    pl.figure()
    pl.plot(theta_range, derrordtheta_euler[0, :], 'r', label='dError/dTheta with Euler dvdtheta')
    pl.plot(theta_range, derrordtheta_quotient, 'b', label='numerical dError/dTheta')
    pl.legend()
    pl.show()
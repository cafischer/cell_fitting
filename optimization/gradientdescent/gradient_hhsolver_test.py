from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from hodgkinhuxley_model.hh_solver import HHSolver
from scipy.interpolate import InterpolatedUnivariateSpline
from create_model import get_model

__author__ = 'caro'


def gradient(theta, v_star, t, v, dvdtheta):

    # compute error for each parameter (theta)
    derrordtheta = np.zeros(len(theta))
    error = np.zeros(len(theta))

    for j in range(len(theta)):

        derrordtheta[j] = 1.0/len(t) * np.sum((v - v_star(t)) * dvdtheta[j])

        error[j] = 1.0/len(t) * np.sum(0.5 * (v - v_star(t))**2)

    return derrordtheta, error


def gradientdescent(theta_init, learn_rate, gradient, num_iterations, *args):

    theta = theta_init
    for i in range(num_iterations):
        theta = theta - learn_rate * gradient(theta, *args)
    return theta


if __name__ == '__main__':

    cell = get_model()

    # create odesolver
    data_dir = './testdata/modeldata_nafka2.csv'
    data = pd.read_csv(data_dir)

    v_star = InterpolatedUnivariateSpline(np.array(data.t), np.array(data.v), k=3)
    t = np.array(data.t)
    v0 = v_star(0)
    i_inj = np.array(data.i)
    y0 = 0
    hhsolver = HHSolver()
    dtheta = 0.01
    theta_max = 0.1
    theta_idx = 0
    theta_range = np.arange(0, theta_max+dtheta, dtheta)

    v = np.zeros(len(theta_range), dtype=object)
    y = np.zeros(len(theta_range), dtype=object)

    for i, theta in enumerate(theta_range):
        cell.ionchannels[theta_idx].g_max = theta
        v[i], _, _, y[i] = hhsolver.solve_adaptive_y(cell, t, v_star, v0, y0, theta_idx)

    # numerical dvdtheta
    #dvdtheta_quotient = np.zeros((len(theta_range), len(t)))
    #for ts in range(len(t)):
    #    v_ts = np.array([v[i][ts] for i in range(len(theta_range))])
    #    dvdtheta_quotient[:, ts] = np.gradient(v_ts, dtheta)

    # compare y and numerical dvdtheta
    #for i, theta in enumerate(theta_range):
    #    pl.figure()
    #    pl.plot(t, dvdtheta_quotient[i, :], 'k', label='num. dvdtheta')
    #    pl.plot(t, y[i], 'r', label='y')
    #    pl.legend()
    #    pl.show()

    # compare numerical derrordtheta with derrordtheta with Euler dvdtheta
    derrordtheta = np.zeros((2, len(theta_range)))
    error = np.zeros((2, len(theta_range)))
    g = np.zeros(len(cell.ionchannels))
    dvdtheta = np.zeros(len(cell.ionchannels), dtype=object)
    for i, theta in enumerate(theta_range):
        g[theta_idx] = theta
        dvdtheta[theta_idx] = y[i]
        derrordtheta[:, i], error[:, i] = gradient(g, v_star, t, v[i], dvdtheta)

    derrordtheta_quotient = np.gradient(error[0, :], dtheta)

    pl.figure()
    pl.plot(theta_range, derrordtheta[theta_idx, :], 'r', label='dError/dTheta')
    pl.plot(theta_range, derrordtheta_quotient, 'b', label='numerical dError/dTheta')
    pl.legend()
    pl.show()


# TODO: 3D plot of error landscape
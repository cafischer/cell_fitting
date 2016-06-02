from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm
from scipy.interpolate import InterpolatedUnivariateSpline
from hodgkinhuxley_model.hh_solver import HHSolver
from create_model import get_model

__author__ = 'caro'


def dvdtheta(theta, hhsolver, cell, t, v_star, v0, y0):
    for i in range(len(theta)):
        cell.ionchannels[i].g_max = theta[i]
    y = np.zeros(len(theta), dtype=object)
    for i in range(len(theta)):
        v, _, _, y[i] = hhsolver.solve_adaptive_y(cell, t, v_star, v0, y0, i)
    return v, y

def gradient(theta, hhsolver, cell, t, v_star, v0, y0):

    v, y = dvdtheta(theta, hhsolver, cell, t, v_star, v0, y0)

    # compute error for each parameter in theta
    derrordtheta = np.zeros(len(theta))
    for i in range(len(theta)):
        derrordtheta[i] = np.sum((v - v_star(t)) * y[i]) / len(t)
    return derrordtheta

if __name__ == '__main__':

    # make model
    cell = get_model()

    # create odesolver
    data_dir = './testdata/modeldata_nafka2.csv'
    data = pd.read_csv(data_dir)

    v_star = InterpolatedUnivariateSpline(np.array(data.t), np.array(data.v), k=3)
    t = np.array(data.t)
    v0 = v_star(0)
    y0 = 0
    hhsolver = HHSolver()
    args = (hhsolver, cell, t, v_star, v0, y0)

    dtheta = 0.01
    theta_max = 1
    theta_range = np.arange(0, theta_max+dtheta, dtheta)

    n_thetas = 2
    derrordtheta = np.zeros(n_thetas, dtype=object)
    for i in range(n_thetas):
        derrordtheta[i] = np.zeros((len(theta_range), len(theta_range)))

    for i, theta0 in enumerate(theta_range):
        for j, theta1 in enumerate(theta_range):
            e_tmp = gradient([theta0, theta1], *args)
            for k in range(n_thetas):
                derrordtheta[k][i, j] = e_tmp[k]

    # TODO: save error

for i in range(n_thetas):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_xlim3d([0.0, 1.0])
    ax.set_xlabel('$g_{naf}$')
    ax.set_ylim3d([0.0, 1.0])
    ax.set_ylabel('$g_{ka}$')
    ax.set_zlabel('$derror/dtheta$')
    ax.view_init(elev=30, azim=50)
    X, Y = np.meshgrid(theta_range, theta_range)
    surf = ax.plot_surface(X, Y, derrordtheta[i].T, rstride=1, cstride=1, cmap=cm.coolwarm, alpha=0.6,
                       linewidth=0, antialiased=False)
    fig.colorbar(surf)
    plt.show()
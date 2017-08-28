import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as pl
from matplotlib.pyplot import cm as cmap
from cell_fitting.optimization.helpers import convert_from_unit

__author__ = 'caro'


def linear_regression(dvdt, i_inj, currents, i_pas=0, Cm=None, cell_area=None):
    if Cm is None:  # estimate cm in the linear regression (in this case we need the cell_area)
        # variables to fit
        i_inj -= i_pas
        X = np.c_[-1 * np.vstack(currents).T, i_inj.T]
        y = dvdt

        # linear regression
        weights, residual = nnls(X, y)
        Cm = 1/weights[-1]
        weights_adjusted = weights * Cm
        weights_adjusted[-1] = convert_from_unit('h', Cm / cell_area)  # uF/cm**2

        return weights_adjusted, weights, residual, y, X
    else:
        # variables to fit
        X = -1 * np.vstack(currents).T
        y = dvdt * Cm - i_inj + i_pas

        # linear regression
        weights, residual = nnls(X, y)

        return weights, residual, y, X


def plot_fit(y, X, weights, t, channel_list, i_pas=0, save_dir=None):

    pl.figure()
    pl.plot(t, y, 'k', label='$c_m \cdot dV/dt - i_{inj} + i_{pas}$'
            if np.any(i_pas != 0) else '$c_m \cdot dV/dt - i_{inj}$')
    pl.plot(t, np.dot(X, weights).T, 'r', label='$-g \cdot \sum_{ion} i_{ion}$')
    pl.plot(t, np.zeros(len(y)), c='0.5')
    pl.ylabel('Current (A)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.legend(loc='upper right', fontsize=16)
    pl.tight_layout()
    if save_dir is not None: pl.savefig(save_dir+'bestfit_derivative.png')
    pl.show()

    # plot current trace and derivative
    pl.figure()
    color = iter(cmap.gist_rainbow(np.linspace(0, 1, len(channel_list))))
    pl.plot(t, y, 'k', label='$c_m \cdot dV/dt - i_{inj} + i_{pas}$'
            if np.any(i_pas != 0) else '$c_m \cdot dV/dt - i_{inj}$')
    for j, current in enumerate(channel_list):
        pl.plot(t, (X.T[j]*weights[j]).T, c=next(color), label=channel_list[j])
    pl.plot(t, np.zeros(len(y)), c='0.5')
    pl.ylabel('Current (A)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.legend(loc='upper right', fontsize=16)
    pl.tight_layout()
    if save_dir is not None: pl.savefig(save_dir+'bestfit_currents.png')
    pl.show()


def sort_weights_by_variable_keys(channel_list, weights, variable_keys):
    channel_names = [k[0][2] for k in variable_keys]
    channel_dict = {channel_names[i]: i for i in range(len(channel_names))}
    channel_list_ordered = [channel_dict[c] for c in channel_list]
    sorted_weights = [c for (n, c) in sorted(zip(channel_list_ordered, weights))]
    return sorted_weights
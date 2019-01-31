import matplotlib.pyplot as pl
import numpy as np
import os
from matplotlib.pyplot import cm as cmap
from scipy.optimize import nnls
from cell_fitting.util import convert_from_unit
pl.style.use('paper_subplots')

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
    if save_dir is not None: pl.savefig(os.path.join(save_dir, 'bestfit_derivative.png'))

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
    if save_dir is not None: pl.savefig(os.path.join(save_dir, 'bestfit_currents.png'))


def plot_fit2(y, X, weights, t_exp, v_exp, t_fit, v_fit, channel_list, i_pas=0, save_dir=None):

    fig, axes = pl.subplots(3, 1, figsize=(8, 8), sharex=True, gridspec_kw={'wspace': 0.1})

    # sum of currents
    axes[0].plot(t_exp, y, 'k', label=r'$c_m \cdot \frac{dV}{dt} - I_{inj} + I_{Leak}$'
                if np.any(i_pas != 0) else r'$c_m \cdot \frac{dV}{dt} - I_{inj}$')
    axes[0].plot(t_exp, np.zeros(len(y)), c='0.5', linestyle='--')
    axes[0].plot(t_exp, np.dot(X, weights).T, 'r', label='$-g_{max} \cdot \sum_{ion\ channel} I_{ion\ channel}$')
    axes[0].set_ylabel('Current (A)')
    axes[0].set_ylim(-150, 150)
    axes[0].legend(loc='upper right')
    axes[0].text(-0.115, 1.0, 'A', transform=axes[0].transAxes, size=18, weight='bold')

    # current traces
    colors = ['y', 'xkcd:orange', 'xkcd:red', 'm', 'g', 'c', 'b', 'sienna']
    #color = iter(cmap.gist_rainbow(np.linspace(0, 1, len(channel_list))))
    axes[1].plot(t_exp, y, 'k', label=r'$c_m \cdot \frac{dV}{dt} - I_{inj} + I_{Leak}$'
            if np.any(i_pas != 0) else r'$c_m \cdot \frac{dV}{dt} - I_{inj}$')
    for j, current in enumerate(channel_list):
        channel_name = r'$-g_{max} \cdot I_{'\
                       + channel_list[j].replace('_', '\ ').replace('sh', 'SH').replace('j', 'J').replace('pas', 'Leak')\
                       + '}$'
        axes[1].plot(t_exp, (X.T[j]*weights[j]).T, c=colors[j], label=channel_name)
    axes[1].plot(t_exp, np.zeros(len(y)), c='0.5')
    axes[1].set_ylabel('Current (A)')
    axes[1].set_ylim(-150, 150)
    axes[1].legend(loc='lower right')
    axes[1].text(-0.115, 1.0, 'B', transform=axes[1].transAxes, size=18, weight='bold')

    # voltage trace
    axes[2].plot(t_exp, v_exp, 'k', label='Data')
    axes[2].plot(t_fit, v_fit, 'r', label='$Model_{linear\ reg.}$')
    axes[2].set_ylabel('Mem. pot. (mV)')
    axes[2].set_xlabel('Time (ms)')
    axes[1].set_xlim(8, 60)
    axes[2].legend(loc='upper right')
    axes[2].text(-0.115, 1.0, 'C', transform=axes[2].transAxes, size=18, weight='bold')

    pl.tight_layout()
    pl.subplots_adjust(top=0.97, right=0.98, bottom=0.06, left=0.11)
    if save_dir is not None: pl.savefig(os.path.join(save_dir, 'linear_regression_stellate_channels.png'))
    pl.show()



def sort_weights_by_variable_keys(channel_list, weights, variable_keys):
    channel_names = [k[0][2] for k in variable_keys]
    channel_dict = {channel_names[i]: i for i in range(len(channel_names))}
    channel_list_ordered = [channel_dict[c] for c in channel_list]
    sorted_weights = [c for (n, c) in sorted(zip(channel_list_ordered, weights))]
    return sorted_weights


def update_variables(cell, variable_values, variable_keys):
    for i in range(len(variable_values)):
            for path in variable_keys[i]:
                cell.update_attr(path, variable_values[i])
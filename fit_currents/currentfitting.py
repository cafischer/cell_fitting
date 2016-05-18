from __future__ import division
import numpy as np
import pandas as pd
import pylab as pl
import scipy.optimize
from model.cell_builder import *
import optimization
from sklearn import linear_model
from matplotlib.pyplot import cm as cmap

def derivative_ridgeregression(y, dt, alpha):
    X = np.tril(dt * np.ones((len(y), len(y))))
    clf = linear_model.Ridge(alpha=alpha)
    clf.fit(X, y)
    dy = clf.coef_
    return dy


def current_fitting(dvdt, t, i_inj, channel_currents, cell_area, channel_list, cm=1, i_passive=0, fit_cm=False,
                    save_dir=None, plot=False, return_fit=False):

    channel_currents = -1 * 1e3 * cell_area * copy.copy(channel_currents)
    i_inj = 1e-3 * copy.copy(i_inj)

    if fit_cm:
        # variables to fit
        i_inj -= i_passive
        X = np.array(np.vstack((channel_currents, i_inj))).T
        y = copy.copy(dvdt)

        # linear regression
        weights, residual = scipy.optimize.nnls(X, y)
        Cm = 1/weights[-1]
        best_fit = {channel_list[i]: weights[i]*Cm for i in range(len(weights)-1)}
        best_fit['cm'] = Cm / cell_area
    else:
        # variables to fit
        Cm = cm * cell_area
        X = np.array(channel_currents).T
        y = copy.copy(dvdt) * Cm - i_inj + i_passive

        # linear regression
        weights, residual = scipy.optimize.nnls(X, y)
        best_fit = {channel_list[i]: weights[i] for i in range(len(weights))}

    if plot:
        # TODO
        #weights[0] = 0.04
        #weights[7] = 1
        #best_fit = {channel_list[i]: weights[i] for i in range(len(weights))}
        # TODO

        pl.figure()
        pl.plot(t, y, 'k', linewidth=1.5, label='$c_m \cdot dV/dt - i_{inj} + i_{pas}$'
                if np.any(i_passive != 0) else '$c_m \cdot dV/dt - i_{inj}$')
        pl.plot(t, np.dot(X, weights), 'r', linewidth=1.5, label='$-g \cdot \sum_{ion} i_{ion}$')
        pl.plot(t, np.zeros(len(y)), c='0.5')

        #pl.plot(t, i_inj, 'b')  #TODO
        #pl.plot(t, i_passive, 'g') # TODO
        pl.ylabel('Current (uA)', fontsize=18)
        pl.xlabel('Time (ms)', fontsize=18)
        pl.legend(loc='upper right', fontsize=18)
        #pl.xlim([10.0, 20.0])  # TODO
        pl.tight_layout()
        if save_dir is not None: pl.savefig(save_dir+'bestfit_derivative.png')
        pl.show()

        # plot current trace and derivative
        pl.figure()
        color = iter(cmap.gist_rainbow(np.linspace(0, 1, len(channel_list))))
        pl.plot(t, y, 'k', linewidth=1.5, label='$c_m \cdot dV/dt - i_{inj} + i_{pas}$'
                if np.any(i_passive != 0) else '$c_m \cdot dV/dt - i_{inj}$')
        for j, current in enumerate(channel_list):
            pl.plot(t, X.T[j]*weights[j], c=next(color), linewidth=1.5, label=channel_list[j])
        pl.plot(t, np.zeros(len(y)), c='0.5')
        pl.ylabel('Current (uA)', fontsize=18)
        pl.xlabel('Time (ms)', fontsize=18)
        pl.legend(loc='upper right', fontsize=18)
        #pl.xlim([10.0, 20.0])  # TODO
        pl.tight_layout()
        if save_dir is not None: pl.savefig(save_dir+'bestfit_currents.png')
        pl.show()

    if return_fit:
        return best_fit, residual, y, np.dot(X, weights)

    return best_fit, residual


def simulate(best_fit, cell, E_ion, data, C_ion=None, onset=0, cut_onset=True, save_dir=None, plot=False):
    # Note: assumes fixed external and internal Ca concentration

    # insert channel and set conductance to 1
    for channel, gbar in best_fit.iteritems():
        cell.update_attr(['soma', 'mechanisms', channel, 'gbar'], gbar)

    # Ca concentration
    if h.ismembrane(str('ca_ion'), sec=cell.soma):
        cell.update_attr(['ion', 'ca_ion', 'cai0'], C_ion['cai'])
        cell.update_attr(['ion', 'ca_ion', 'cao0'], C_ion['cao'])

    # set equilibrium potentials
    cell = set_Epotential(cell, E_ion)

    # extract simulation parameters
    simulation_params = optimization.optimizer.extract_simulation_params(data)
    simulation_params.update({'onset': onset})
    if not cut_onset:
        simulation_params.update({'cut_onset': False})

    # run simulation
    v_fitted, t_fitted = optimization.fitfuns.run_simulation(cell, **simulation_params)

    # compute error
    if not cut_onset:
        error = optimization.errfuns.rms(v_fitted[onset/(t_fitted[1]-t_fitted[0]):], data.v)
    else:
        error = optimization.errfuns.rms(v_fitted, data.v)

    # plot the results
    if plot:
        pl.figure()
        pl.plot(data.t, data.v, 'k', linewidth=1.5, label='data')
        if cut_onset:
            pl.plot(t_fitted, v_fitted, 'r', linewidth=1.5, label='model')
        else:
            pl.plot(t_fitted - onset, v_fitted, 'r', linewidth=1.5, label='model')
        pl.ylabel('Membrane \npotential (mV)', fontsize=18)
        pl.xlabel('Time (ms)', fontsize=18)
        pl.legend(loc='upper right', fontsize=18)
        if save_dir is not None: pl.savefig(save_dir+'bestfit.png')
        pl.show()

    return error, t_fitted, v_fitted


def set_Epotential(cell, E_ion):
    # set equilibrium potentials
    for eion, E in E_ion.iteritems():
        if eion == "epas":
            if hasattr(cell.soma(.5), 'passive'):
                cell.update_attr(['soma', 'mechanisms', 'passive', eion], E)
        elif eion == 'ehcn':
            if hasattr(cell.soma(.5), 'ih_slow'):
                cell.update_attr(['soma', 'mechanisms', 'ih_slow', eion], E)
            if hasattr(cell.soma(.5), 'ih_fast'):
                cell.update_attr(['soma', 'mechanisms', 'ih_fast', eion], E)
        elif eion == 'ekleak':
            if hasattr(cell.soma(.5), 'kleak'):
                cell.update_attr(['soma', 'mechanisms', 'kleak', eion], E)
        else:
            cell.update_attr(['ion', eion[1:]+'_ion', eion], E)
    return cell

# TODO: divide simulation and error calculation
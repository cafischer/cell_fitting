from __future__ import division
import numpy as np
import pandas as pd
import pylab as pl
import scipy.optimize
import os
import sys
from model.cell_builder import *
import optimization
import fit_currents
from sklearn import linear_model
import json_utils
import copy


def derivative_ridgeregression(y, dt, alpha):
    X = np.tril(dt * np.ones((len(y), len(y))))
    clf = linear_model.Ridge(alpha=alpha)
    clf.fit(X, y)
    dy = clf.coef_
    return dy


def current_fitting(dvdt, t, i_inj, channel_currents, cell_area, channel_list, cm, i_passive=0, fit_cm=False,
                    save_dir=None, plot=False):

    # unit conversion and sign
    for i in range(len(channel_currents)):
        channel_currents[i] *= -1 * 1e3 * cell_area
    i_inj *= 1e-3

    if fit_cm:
        # variables to fit
        i_inj -= i_passive
        X = np.array(np.vstack((channel_currents, i_inj))).T
        y = dvdt

        # linear regression
        weights, residual = scipy.optimize.nnls(X, y)
        Cm = 1/weights[-1]
        best_fit = {channel_list[i]: weights[i]*Cm for i in range(len(weights)-1)}
        best_fit['cm'] = Cm / cell_area
    else:
        # variables to fit
        Cm = cm * cell_area
        X = np.array(channel_currents).T
        y = dvdt * Cm - i_inj + i_passive

        # linear regression
        weights, residual = scipy.optimize.nnls(X, y)
        best_fit = {channel_list[i]: weights[i] for i in range(len(weights))}

    # print best fit
    print "Best fit: "
    print best_fit
    print "Residual: " + str(residual)

    if plot:
        pl.figure()
        pl.plot(t, y, 'k', linewidth=1.5, label='$c_m \cdot dV/dt - i_{inj} + i_{pas}$'
                if np.any(i_passive != 0) else '$c_m \cdot dV/dt - i_{inj}$')
        pl.plot(t, np.dot(X, weights), 'r', linewidth=1.5, label='$-g \cdot \sum_{ion} i_{ion}$')
        pl.ylabel('Current (pA)', fontsize=18)
        pl.xlabel('Time (ms)', fontsize=18)
        #pl.xlim([5, 30])
        pl.legend(loc='upper right', fontsize=18)
        pl.savefig(save_dir+'bestfit_derivative.png')
        pl.show()

        # plot current trace and derivative
        pl.figure()
        pl.plot(t, y, 'k', linewidth=1.5, label='$c_m \cdot dV/dt - i_{inj} + i_{pas}$'
                if np.any(i_passive != 0) else '$c_m \cdot dV/dt - i_{inj}$')
        for j, current in enumerate(channel_list):
            pl.plot(t, X.T[j]*weights[j], linewidth=1.5, label=channel_list[j])
        pl.ylabel('Current (pA)', fontsize=18)
        pl.xlabel('Time (ms)', fontsize=18)
        #pl.xlim([5, 30])
        pl.legend(loc='upper right', fontsize=18)
        pl.savefig(save_dir+'currents.png')
        pl.show()

    return best_fit, residual


def simulate(best_fit, cell, E_ion, data, save_dir, plot=False):

    # update conductances
    for var, val in best_fit.iteritems():
        if var == 'cm':
            cell.update_attr(["soma", "cm"], val)
        elif var == 'passive':
            cell.update_attr(["soma", "mechanisms", "pas"], {})
            cell.update_attr(["ion", "pas", "g_pas"], val)
        elif var == 'ih_fast':
            cell.update_attr(["soma", "mechanisms", "ih", "gfastbar"], val)
        elif var == 'ih_slow':
            cell.update_attr(["soma", "mechanisms", "ih", "gslowbar"], val)
        elif var == 'caLVA':
            cell.update_attr(["soma", "mechanisms", var, "pbar"], val)
        else:
            cell.update_attr(["soma", "mechanisms", var, "gbar"], val)

    # set equilibrium potentials
    for eion, E in E_ion.iteritems():
        if eion == 'ehcn':
            for seg in cell.soma:
                seg.ih.ehcn = E
        elif eion == 'ekleak':
            for seg in cell.soma:
                seg.kleak.ekleak = E
        else:
            setattr(cell.soma, eion, E)

    # insert calcium pump if calcium is present
    if h.ismembrane('ca_ion', sec=cell.soma):
        Mechanism('cad').insert_into(cell.soma)

    # extract simulation parameters
    objective=''  #TODO
    simulation_params = optimization.optimizer.extract_simulation_params([objective], {objective: data})
    simulation_params.update({'onset': {objective: 200}})
    simulation_params_tmp = {p: simulation_params[p][objective] for p in simulation_params}

    # run simulation
    v_fitted, t = optimization.fitfuns.run_simulation(cell, **simulation_params_tmp)

    # compute error
    error = optimization.optimizer.mean_squared_error(v_fitted, data.v)
    print "Error: " + str(error)

    # plot the results
    if plot:
        pl.figure()
        pl.plot(t, data.v, 'k', linewidth=2, label='data')
        pl.plot(t, v_fitted, 'r', linewidth=2, label='model')
        pl.ylabel('Membrane \npotential (mV)', fontsize=18)
        pl.xlabel('Time (ms)', fontsize=18)
        pl.legend(loc='upper right', fontsize=18)
        pl.savefig(save_dir+'bestfit.png')
        pl.show()

    return error, t, v_fitted
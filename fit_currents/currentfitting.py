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


def current_fitting(dv, i_inj, channel_currents, cell_area, channel_list, fit_cm=False):

    # unit conversion and sign
    for i in range(len(channel_currents)):
        channel_currents[i] *= -1 * 1e3 * cell_area
    i_inj *= 1e-3

    if fit_cm:
        # variables to fit
        X = np.array(np.vstack((channel_currents, i_inj))).T
        y = dv

        # linear regression
        weights, residual = scipy.optimize.nnls(X, y)
        cm = 1/weights[-1]
        best_fit = {channel_list[i]: weights[i]*cm for i in range(len(weights)-1)}
        best_fit['cm'] = cm
    else:
        # variables to fit
        Cm = cell.soma.cm * cell_area
        X = np.array(channel_currents).T
        y = dv * Cm - i_inj

        # linear regression
        weights, residual = scipy.optimize.nnls(X, y)
        best_fit = {channel_list[i]: weights[i] for i in range(len(weights))}

    # print best fit
    print "Best fit: "
    print best_fit
    print "Residual: " + str(residual)

    # TODO
    pl.figure()
    pl.plot(t, y, 'k', linewidth=1.5, label='y')
    pl.plot(t, np.dot(X, weights), 'b', linewidth=1.5, label='y fitted')
    pl.legend()
    pl.show()
    # TODO

    return best_fit, residual


def simulate(best_fit, cell, E_ion, data, plot=False):

    # update conductances
    for var, val in best_fit.iteritems():
        if var == 'cm':
            cell.soma.cm = val
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
        else:
            setattr(cell.soma, eion, E)

    # insert calcium pump if calcium is present
    if h.ismembrane('ca_ion', sec=cell.soma):
        Mechanism('cad').insert_into(cell.soma)

    # extract simulation parameters
    simulation_params = optimization.optimizer.extract_simulation_params([objective], {objective: data})
    simulation_params.update({'onset': {objective: 200}})
    simulation_params_tmp = {p: simulation_params[p][objective] for p in simulation_params}

    # run simulation
    v_fitted, t = optimization.fitfuns.run_simulation(cell, **simulation_params_tmp)

    # compute error
    error = optimization.optimizer.quadratic_error(v_fitted, data.v)
    print "Error: " + str(error)

    # plot the results
    if plot:
        pl.figure()
        pl.plot(t, v, 'k', linewidth=2, label='data')
        pl.plot(t, v_fitted, 'r', linewidth=2, label='model')
        pl.xlabel('Membrane potential (mV)')
        pl.ylabel('Time (ms)')
        pl.legend(loc='lower right')
        pl.title('Best fit')
        pl.show()

    return error

if __name__ == "__main__":

    # parameters
    cellid = '2015_08_11d'
    objective = 'dap'
    save_dir = './fit_'+cellid+'/'+objective+'/'
    data_dir = '../data/new_cells/'+cellid+'/dap/dap.csv'
    current_dir = './current_traces/'+cellid+'/'+objective+'/'
    #model_dir = '../model/cells/point.json'
    model_dir = './fit_'+cellid+'/stepcurrent/cell.json'
    mechanism_dir = '../model/channels'
    mechanism_dir_clamp = '../model/channels_without_output'

    channel_list = ['nav16', 'km', 'caHVA', 'caLVA', 'kca', 'kdr', 'ka', 'na8st', 'narsg', 'nat', 'nap'] # ['passive', 'ih_slow', 'ih_fast']
    E_ion = {'ek': -83, 'eca': 90, 'ena': 87}  # {'ehcn': -20, 'e_pas': -73}

    keys = ["soma", "mechanisms", "na8st", "vshift"]
    var_name = keys[-1]
    var_range = np.arange(-20, 21, 5)

    n_chunks = 2  #10  # for derivative
    alpha = 0.5

    # load mechanisms
    if sys.maxsize > 2**32: mechanism_dir += '/x86_64/.libs/libnrnmech.so'
    else: mechanism_dir += '/i686/.libs/libnrnmech.so'
    h.nrn_load_dll(mechanism_dir)
    if sys.maxsize > 2**32: mechanism_dir_clamp += '/x86_64/.libs/libnrnmech.so'
    else: mechanism_dir_clamp += '/i686/.libs/libnrnmech.so'
    h.nrn_load_dll(mechanism_dir_clamp)

    # load data
    data = pd.read_csv(data_dir)
    v = np.array(data.v)
    i_inj = np.array(data.i)
    t = np.array(data.t)

    # estimate derivative via ridge regression
    dt = t[1] - t[0]
    chunk = len(t) / n_chunks
    ds = np.DataSource()
    if ds.exists(save_dir+'/dv.npy'):
        with open(save_dir+'/dv.npy', 'r') as f:
            dv = np.load(f)
    else:
        dv = np.zeros(len(t))
        for i in range(int(len(t)/chunk)):
            dv[i*chunk:(i+1)*chunk] = derivative_ridgeregression(v[i*chunk:(i+1)*chunk], dt, alpha)
        pl.figure()
        pl.plot(t, dv, 'k', label='alpha: '+str(alpha) + '\n' + 'n_chunks: ' + str(n_chunks))
        pl.legend()
        pl.ylabel('dV/dt (mV/ms)')
        pl.xlabel('Time (ms)')
        pl.show()

    # change parameters and find best fit
    best_fit = [[]] * len(var_range)
    errors = [[]] * len(var_range)

    for i, val in enumerate(var_range):
        print var_name + ': ' + str(val)

        # create cell
        cell = Cell(model_dir)
        cell_area = cell.soma(.5).area() * 1e-8

        # update cell
        cell.update_attr(keys, val)
        #E_ion[var_name] = val

        # compute currents
        channel_currents = fit_currents.vclamp.vclamp(v, t, cell, channel_list, E_ion)

        # linear regression
        best_fit[i], residual = current_fitting(dv, copy.deepcopy(i_inj), channel_currents, cell_area, channel_list)

        # run simulation to check best fit
        errors[i] = simulate(best_fit[i], cell, E_ion, data, plot=False)

        # plot current trace and derivative
        """
        for j, current in enumerate(channel_currents):
            #f, (ax1, ax2) = pl.subplots(2, 1, sharex=True)
            #ax1.plot(t, dv, 'k', linewidth=2)
            #ax2.plot(t, current, 'k', linewidth=2)
            #ax2.set_xlabel('Time (ms)', fontsize=18)
            #ax1.set_ylabel('dV/dt (mV/ms)', fontsize=18)
            #ax2.set_ylabel('Current (nA)', fontsize=18)
            #pl.show()
            pl.figure()
            pl.plot(t, dv, 'k', linewidth=1.5, label='dv/dt')
            pl.plot(t, current*np.max(np.abs(dv))/np.max(np.abs(current)), linewidth=1.5, label=channel_list[j])
            pl.legend()
            pl.show()
        """

    # find best fit
    best = np.argmin(errors)

    print
    print "Best fit over parameter: "
    print var_range[best]
    print best_fit[best]
    print "Error: " + str(errors[best])

    # plot best fit
    cell = Cell(model_dir)
    cell.update_attr(keys, var_range[best])
    simulate(best_fit[best], cell, E_ion, data, plot=True)

    # save results
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir+'/best_fit.json', 'w') as f:
        json_utils.dump(best_fit[best], f)
    np.savetxt(save_dir+'/error.json', np.array([errors[best]]))
    cell.save_as_json(save_dir+'cell.json')
    with open(save_dir+'/dv.npy', 'w') as f:
        np.save(f, dv)


# TODO save E in cell
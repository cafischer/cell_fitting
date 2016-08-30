import json
import os

import numpy as np
from nrn_wrapper import Cell

from optimization.simulate import currents_given_v
from optimization.problems import CellFitProblem, get_channel_list, get_ionlist, convert_units
from optimization.linear_regression import linear_regression, plot_fit

__author__ = 'caro'

# TODO
from neuron import h
from optimization.problems import complete_mechanismdir
m_dir = '../../../model/channels/icgenealogy/Kchannels'
h.nrn_load_dll(complete_mechanismdir(m_dir))
m_dir = '../../../model/channels/icgenealogy/Nachannels'
h.nrn_load_dll(complete_mechanismdir(m_dir))

# parameter
save_dir = '../../../results/linear_regression/dapmodelnaka/'
n_trials = 1

params = {
          'name': 'CellFitProblem',
          'maximize': False,
          'normalize': True,
          'model_dir': '../../../model/cells/dapmodelnaka.json',
          'mechanism_dir': '../../../model/channels/schmidthieber',
          'variables': [],
          'data_dir': '../../../data/2015_08_11d/merged/step_dap_zap.csv',
          'get_var_to_fit': 'get_v',
          'fitnessweights': [1.0],
          'errfun': 'rms',
          'insert_mechanisms': True
         }

# create problem
problem = CellFitProblem(**params)

# save all information
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(save_dir+'/problem.json', 'w') as f:
    json.dump(params, f, indent=4)
with open(save_dir+'/cell.json', 'w') as f:
    json.dump(Cell.from_modeldir(params['model_dir']).get_dict(), f, indent=4)

for trial in range(0, n_trials):

    # get current traces
    v_exp = problem.data.v.values
    t_exp = problem.data.t.values
    i_exp = problem.data.i.values
    dt = t_exp[1] - t_exp[0]
    dvdt = np.concatenate((np.array([(v_exp[1]-v_exp[0])/dt]), np.diff(v_exp) / dt))
    candidate = np.ones(len(problem.path_variables))
    problem.update_cell(candidate)
    channel_list = get_channel_list(problem.cell, 'soma')
    ion_list = get_ionlist(channel_list)
    celsius = problem.simulation_params['celsius']

    currents = currents_given_v(v_exp, t_exp, problem.cell.soma, channel_list, ion_list, celsius)

    # convert units
    dvdt_sc, i_inj_sc, currents_sc, Cm, cell_area = convert_units(problem.cell.soma.L, problem.cell.soma.diam,
                                                          problem.cell.soma.cm, dvdt, i_exp,
                                                          currents)

    # linear regression
    weights, residual, y, X = linear_regression(dvdt_sc, i_inj_sc, currents_sc, i_pas=0, Cm=Cm)
    #weights, residual, y, X = linear_regression(dvdt_sc, i_inj_sc, currents_sc, i_pas=0, Cm=None, cell_area=cell_area)

    # output
    print 'channels: ' + str(channel_list)
    print 'weights: ' + str(weights)

    # plot fit
    # plot in three parts
    merge_points = [0, 22999, 26240, 550528]
    for i in range(len(merge_points)-1):
        y_plot = y[merge_points[i]:merge_points[i+1]]
        X_plot = X[merge_points[i]:merge_points[i+1], :]
        t_plot = t_exp[merge_points[i]:merge_points[i+1]]
        plot_fit(y_plot, X_plot, weights, t_plot, channel_list, save_dir=save_dir+str(i))

    # save
    np.savetxt(save_dir+'/best_candidate_'+str(trial)+'.txt', weights)
    np.savetxt(save_dir+'/error_'+str(trial)+'.txt', np.array([residual]))

    # simulate
    #cm = weights[-1]
    for i, w in enumerate(weights[:]):
        keys = ['soma', '0.5', channel_list[i], 'gbar']
        if channel_list[i] == 'pas':
            keys = ['soma', '0.5', channel_list[i], 'g']
            problem.cell.update_attr(keys, w)
        elif 'ion' in channel_list[i]:
            keys = None
        else:
            keys = ['soma', '0.5', channel_list[i], 'gbar']
            problem.cell.update_attr(keys, w)

    from optimization.simulate import run_simulation
    v, t = run_simulation(problem.cell, **problem.simulation_params)

    import matplotlib.pyplot as pl
    pl.figure()
    pl.plot(t, problem.data.v, 'k')
    pl.plot(t, v, 'r')
    pl.show()
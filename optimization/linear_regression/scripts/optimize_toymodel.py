import numpy as np
import json
import os

from nrn_wrapper import Cell
from optimization.simulate import currents_given_v
from optimization.bio_inspired.problems import CellFitProblem, get_ionlist, convert_units
from optimization.linear_regression import linear_regression, plot_fit

__author__ = 'caro'


# parameter
save_dir = '../../../results/linear_regression/test'
n_trials = 1

variables = [
            [0, 2.5, [['soma', '0.5', 'na8st', 'gbar']]]
            ]

params = {
          'maximize': False,
          'normalize': True,
          'model_dir': '../../../model/cells/toymodel1.json',
          'mechanism_dir': '../../../model/channels/schmidthieber',
          'variables': variables,
          'data_dir': '../../../data/toymodels/toymodel1/ramp_dt.csv',
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
    cell = problem.get_cell(candidate)
    channel_list = list(set([problem.path_variables[i][0][2] for i in range(len(problem.path_variables))]))
        # only works if channel name is at 2 second position in the path!
    ion_list = get_ionlist(channel_list)
    celsius = problem.simulation_params['celsius']

    currents = currents_given_v(v_exp, t_exp, cell.soma, channel_list, ion_list, celsius)

    # convert units
    dvdt_sc, i_inj_sc, currents_sc, Cm, _ = convert_units(cell.soma.L, cell.soma.diam, cell.soma.cm, dvdt, i_exp, currents)

    # linear regression
    weights, residual, y, X = linear_regression(dvdt_sc, i_inj_sc, currents_sc, i_pas=0, Cm=Cm)

    # output
    print 'weights: ' + str(weights)

    # plot fit
    plot_fit(y, X, weights, t_exp, channel_list, save_dir=save_dir)

    # save
    np.savetxt(save_dir+'/best_candidate_'+str(trial)+'.txt', weights)
    np.savetxt(save_dir+'/error_'+str(trial)+'.txt', np.array([residual]))
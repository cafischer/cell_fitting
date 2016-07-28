from random import Random
from time import time
import numpy as np
import os
import json
import matplotlib.pyplot as pl
from nrn_wrapper import Cell
from optimization.bio_inspired.problems import CellFitProblem
from scipy.optimize import fsolve
from optimization.simulate import currents_given_v
from optimization.bio_inspired.problems import get_ionlist, convert_units

__author__ = 'caro'

def solve(candidate):
    # vclamp currents
    channel_list = list(set([problem.path_variables[i][0][2] for i in range(len(problem.path_variables))]))
        # only works if channel name is at 2 second position in the path!
    ion_list = get_ionlist(channel_list)
    celsius = problem.simulation_params['celsius']
    dt = problem.simulation_params['dt']
    t = problem.data.t.values
    v = problem.data.v.values
    i_inj = problem.data.i.values
    cell = problem.get_cell(candidate)

    currents = currents_given_v(v, t, cell.soma, channel_list, ion_list, celsius, plot=False)
    dvdt = np.concatenate((np.array([(v[1]-v[0])/dt]), np.diff(v) / dt))

    # unit conversion
    dvdt_sc, i_inj_sc, currents_sc, Cm, _ = convert_units(cell.soma.L, cell.soma.diam, cell.soma.cm, dvdt, i_inj, currents)
    cmdvdt_sc = dvdt_sc * Cm
    i_ion_sc = np.sum(currents_sc, 0)

    # TODO
    #pl.figure()
    #pl.plot(t, cmdvdt_sc-i_inj_sc, 'k')
    #pl.plot(t, -i_ion_sc, 'b')
    #pl.show()

    # hh equation
    hh_eq = cmdvdt_sc - i_inj_sc + i_ion_sc

    # pick points (e.g. where current traces diverge or during current injection)
    current_start = np.nonzero(i_inj)[0][0]
    n_points = len(problem.path_variables)
    hh_eq_points = hh_eq[current_start:current_start+n_points]

    return hh_eq_points


# parameter
save_dir = '../../../results/hhsolver/test'
n_trials = 50

variables = [
            [0, 2.5, [['soma', '0.5', 'na8st', 'gbar']]]
            ]

params = {
          'maximize': False,
          'normalize': False,
          'model_dir': '../../../model/cells/toymodel1.json',
          'mechanism_dir': '../../../model/channels/schmidthieber',
          'variables': variables,
          'data_dir': '../../../data/toymodels/toymodel1/ramp_dt.csv',
          'get_var_to_fit': 'get_v',
          'fitnessweights': [1.0],
          'errfun': 'rms',
          'insert_mechanisms': True
         }

problem = CellFitProblem(**params)

# change dt
#from data import change_dt
#from optimization.bio_inspired.problems import extract_simulation_params
#dt = problem.simulation_params['dt']
#dt_new = dt / 2
#data = pd.read_csv(params['data_dir'])
#data_new = change_dt(dt_new, data)
#problem.data_to_fit[0] = data_new.v
#problem.simulation_params = extract_simulation_params(data_new)

# save all information
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(save_dir+'/problem.json', 'w') as f:
    json.dump(params, f, indent=4)
with open(save_dir+'/cell.json', 'w') as f:
    json.dump(Cell.from_modeldir(params['model_dir']).get_dict(), f, indent=4)


for trial in range(n_trials):
    # create pseudo random number generator
    if os.path.isfile(save_dir+'/seed'+str(trial)+'.txt'):
        seed = float(np.loadtxt(save_dir+'seed_'+str(trial)+'.txt'))
    else:
        seed = time()
        np.savetxt(save_dir+'/seed_'+str(trial)+'.txt', np.array([seed]))
    prng = Random()
    prng.seed(seed)

    # create candidate
    candidate = problem.generator(prng, None)[0]

    # find solution by setting nonlinear equations to zero
    x = fsolve(solve, candidate)

    # output
    print 'start value: ' + str(candidate)
    print 'end value: ' + str(x)

    # save
    np.savetxt(save_dir+'/best_candidate_'+str(trial)+'.txt', x)



from random import Random
from time import time
import numpy as np
import os
import json
from nrn_wrapper import Cell
from optimization.problems import get_ionlist, convert_units
from optimization import problems
from scipy.optimize import fsolve
from optimization.simulate import currents_given_v

__author__ = 'caro'

def solve(candidate):
    # vclamp currents
    #channel_list = list(set([problem.path_variables[i][0][2] for i in range(len(problem.path_variables))]))
        # only works if channel name is at 2 second position in the path!
    channel_list = ['na8st', 'kdr', 'pas']
    ion_list = get_ionlist(channel_list)
    celsius = problem.simulation_params['celsius']
    dt = problem.simulation_params['dt']
    t = problem.data.t.values
    v = problem.data.v.values
    i_inj = problem.data.i.values
    problem.update_cell(candidate)

    currents = currents_given_v(v, t, problem.cell.soma, channel_list, ion_list, celsius, plot=False)
    dvdt = np.concatenate((np.array([(v[1]-v[0])/dt]), np.diff(v) / dt))

    # unit conversion
    dvdt_sc, i_inj_sc, currents_sc, Cm, _ = convert_units(problem.cell.soma.L, problem.cell.soma.diam,
                                                          problem.cell.soma.cm, dvdt, i_inj, currents)
    cmdvdt_sc = dvdt_sc * Cm
    i_ion_sc = np.sum(currents_sc, 0)

    # hh equation
    hh_eq = cmdvdt_sc - i_inj_sc + i_ion_sc

    # pick points (e.g. where current traces diverge or during current injection)
    current_start = np.nonzero(i_inj)[0][0]
    dist = int(np.round(0.5/dt, 0))
    n_points = len(problem.path_variables)
    points = np.arange(current_start, n_points*dist+dist, dist)
    points = np.array([int(np.round(10.8/dt)), int(np.round(11.35/dt)), int(np.round(11.5/dt)),
                       int(np.round(12.0/dt))])  # TODO depends on data  int(np.round(10.4/dt)),
    hh_eq_points = hh_eq[points]

    # TODO
    #pl.figure()
    #pl.plot(t, cmdvdt_sc-i_inj_sc, 'k')
    #pl.plot(t, -i_ion_sc, 'b')
    #pl.plot(t[points], (cmdvdt_sc-i_inj_sc)[points], 'yo')
    #pl.plot(t[points], -i_ion_sc[points], 'ro')
    #pl.show()

    return hh_eq_points


# parameter
save_dir = '../../../results/test_algorithms/increase_params/5param/hhsolver'
n_trials = 20

variables = [
            [0, 2.5, [['soma', '0.5', 'na8st', 'gbar']]],
            [0, 2.5, [['soma', '0.5', 'kdr', 'gbar']]],
            [0, 2.5, [['soma', '0.5', 'pas', 'g']]],
            [0, 1.0, [['soma', '0.5', 'na8st', 'a3_1']]]
            #[0, 50.0, [['soma', '0.5', 'na8st', 'a3_0']]]
            ]

params = {
          'name': 'CellFitProblem',
          'maximize': False,
          'normalize': False,
          'model_dir': '../../../model/cells/toymodel3.json',
          'mechanism_dir': '../../../model/channels/schmidthieber',
          'variables': variables,
          'data_dir': '../../../data/toymodels/toymodel3/ramp_dt.csv',
          'get_var_to_fit': 'get_v',
          'fitnessweights': [1.0],
          'errfun': 'rms',
          'insert_mechanisms': True
         }

problem = getattr(problems, params['name'])(**params)

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
    candidate = problem.generator(prng, None)  # TODO: for 1 param: [0]

    # find solution by setting nonlinear equations to zero
    x = fsolve(solve, candidate)

    # output
    print 'start value: ' + str(candidate)
    print 'end value: ' + str(x)

    # save
    np.savetxt(save_dir+'/best_candidate_'+str(trial)+'.txt', x)

    # TODO: save points used for solving HH
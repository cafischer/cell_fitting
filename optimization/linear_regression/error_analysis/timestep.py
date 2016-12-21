from time import time
from random import Random
import numpy as np
import matplotlib.pyplot as pl
from data import change_dt
from optimization.simulate import extract_simulation_params, run_simulation
from optimization.errfuns import rms
from optimization.simulate import currents_given_v
from optimization.problems import CellFitProblem, get_channel_list, get_ionlist, convert_units
from optimization.linear_regression import linear_regression

__author__ = 'caro'

# parameter
save_dir = '../../../results/linear_regression/error_analysis/timestep/'
n_models = 4
dt_fac = np.arange(0, 5, 1)

variables = [
            [0, 2.5, [['soma', '0.5', 'na8st', 'gbar']]]
            ]

params = {
          'name': 'CellFitProblem',
          'maximize': False,
          'normalize': True,
          'model_dir': '../../../model/cells/toymodel1.json',
          'mechanism_dir': '../../../model/channels/schmidthieber',
          'variables': variables,
          'data_dir': '../../../data/toymodels/toymodel1/ramp.csv',
          'get_var_to_fit': 'get_v',
          'fitnessweights': [1.0],
          'errfun': 'rms',
          'insert_mechanisms': True
         }

# create problem
problem = CellFitProblem(**params)
dt_exp = problem.simulation_params['dt']
dts = dt_exp / 2**dt_fac
channel_list = get_channel_list(problem.cell, 'soma')
ion_list = get_ionlist(channel_list)

# initialize errors
error_weights = np.zeros([n_models, len(dts)])
error_traces = np.zeros([n_models, len(dts)])

# create pseudo random number generator
seed = time()
#np.savetxt(save_dir+'/seed_'+str(trial)+'.txt', np.array([seed]))
prng = Random()
prng.seed(seed)

# generate random models
for i in range(n_models):

    # create a model randomly from all channels
    weights_model = problem.generator(prng, None)

    for j, dt in enumerate(dts):
        print 'run '+'model: ' + str(i) + ' dt: ' +str(dt)
        
        # change dt
        data_newdt = change_dt(dt, problem.data)
        problem.simulation_params = extract_simulation_params(data_newdt)

        # run simulation
        problem.update_cell(weights_model)
        currents = [problem.cell.soma.record_from(channel_list[k], 'i'+ion_list[k], pos=.5) for k in range(len(channel_list))]
        v_newdt, t_newdt = run_simulation(problem.cell, **problem.simulation_params)

        #pl.figure()
        #pl.plot(t_newdt, v_newdt)
        #pl.show()

        # compute parameter
        dvdt_newdt = np.concatenate((np.array([(v_newdt[1]-v_newdt[0])/dt]), np.diff(v_newdt)/dt))
        i_newdt = data_newdt.i.values
        celsius = problem.simulation_params['celsius']

        # get currents
        candidate = np.ones(len(problem.path_variables))  # gbars should be 1
        problem.update_cell(candidate)
        currents_newdt = currents_given_v(v_newdt, t_newdt, problem.cell.soma, channel_list, ion_list, celsius)

        # convert units
        dvdt_sc, i_inj_sc, currents_sc, Cm, _ = convert_units(problem.cell.soma.L, problem.cell.soma.diam,
                                                              problem.cell.soma.cm, dvdt_newdt,
                                                              i_newdt, currents_newdt)

        # linear regression
        weights, residual, y, X = linear_regression(dvdt_sc, i_inj_sc, currents_sc, i_pas=0, Cm=Cm)

        # plots
        #plot_fit(y, X, weights, t_newdt, channel_list)

        # compute error
        error_traces[i, j] = rms(y, np.sum(np.array(currents), 0))

        error_tmp = [rms(weights_model[k], weights[k]) for k in range(len(weights_model))]
        error_weights[i, j] = np.mean(error_tmp)


# compute mean error of all models
print
for j, dt in enumerate(dts):
    print 'Mean error in the weights with dt = '+str(dt)+' ms: ' \
          + str(np.mean(error_weights[:, j]))

print
for j, dt in enumerate(dts):
    print 'Mean error in the fit with dt = '+str(dt)+' ms: ' \
          + str(np.mean(error_traces[:, j]))

# plot
pl.figure()
pl.plot(np.array(dts), np.sum(error_traces, 0), 'ok')
pl.xlim(np.max(dts), 0)
pl.xlabel('$Time step (ms)$', fontsize=16)
pl.ylabel('$Rms(c_m \cdot dV/dt - i_{inj}, -\sum_{ion} i_{ion})$', fontsize=16)
pl.tight_layout()
pl.show()
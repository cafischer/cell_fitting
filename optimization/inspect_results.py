from __future__ import division
import numpy as np
from neuron import h
from model.cell_builder import *
import json_utils
from optimizer import quadratic_error
import optimization
h.load_file("stdrun.hoc")  # load NEURON libraries
h("""cvode.active(0)""")  # invariable time step in NEURON

from optimize_active.optimize_activepointaxon import *

__author__ = 'caro'


if __name__ == "__main__":
    save_dir = './optimize_active/results_active/point/test'
    n_ind = 1

    # load the jsonized optimizer
    with open(save_dir+'/optimizer.json', 'r') as file:
        optimizer_json = json_utils.load(file)

    # get absolute directories for this machine
    for key in optimizer_json.keys():
        if 'dir' in key:
            if np.any([obj in optimizer_json[key] for obj in optimizer_json['objectives']]):
                for obj in optimizer_json['objectives']:
                    optimizer_json[key][obj] = str(os.path.join(os.path.abspath('..'), optimizer_json[key][obj]))
            else:
                optimizer_json[key] = str(os.path.join(os.path.abspath('..'), optimizer_json[key]))

    # complete mechanism_dir with respect to this machine
    if sys.maxsize > 2**32:
        optimizer_json['mechanism_dir'] = optimizer_json['mechanism_dir'] + '/x86_64/.libs/libnrnmech.so'
    else:
        optimizer_json['mechanism_dir'] = optimizer_json['mechanism_dir'] + '/i686/.libs/libnrnmech.so'

    # create Cell
    cell = Cell(optimizer_json['model_dir'], optimizer_json['mechanism_dir'])

    # load data
    data = dict()
    for obj in optimizer_json['objectives']:
        data[obj] = pd.read_csv(os.path.join(os.path.abspath('..'), optimizer_json['data_dir'][obj]))

    # look at individuals
    error = np.zeros(n_ind, dtype=object) # save error of each individual for all objectives (for comparison)
    for ind in range(0, n_ind):
        error[ind] = [0] * (len(optimizer_json['objectives'])+1)

        # update the cell with variables from the best individuals
        with open(save_dir + '/' + 'variables_new_'+str(ind)+'.json', 'r') as file:
            variables_new = json.load(file)
        for i, p in enumerate(variables_new):
            for path in variables_new[i][2]:
                cell.update_attr(path, variables_new[i][1])
        if optimizer_json['recalculate_variables'] is not None:
            variables_new_dict = dict()
            for var in variables_new:
                variables_new_dict[var[0]] = var[1]
            recalculate_variables = getattr(optimization.fitfuns, optimizer_json['recalculate_variables'])
            recalculate_variables(variables_new_dict)

        # run simulation of best individual
        for i, obj in enumerate(optimizer_json['objectives']):
            simulation_params_tmp = dict()
            for p in optimizer_json['simulation_params']:
                simulation_params_tmp[p] = optimizer_json['simulation_params'][p][obj]

            # run simulation and compute the variable to fit
            fun_to_fit = getattr(optimization.fitfuns,
                                 str(optimizer_json['fun_to_fit'][obj]))
            var_fitted, x = fun_to_fit(cell, **simulation_params_tmp)

            # data to fit
            data_to_fit = np.array(data[obj][optimizer_json['var_to_fit'][obj]])  # convert to array
            data_to_fit = data_to_fit[~np.isnan(data_to_fit)]  # get rid of nans

            # compute error
            error[ind][i] = quadratic_error(var_fitted, data_to_fit)

            # plot the results
            pl.figure()
            pl.plot(x, data_to_fit, 'k', linewidth=2, label='data')
            pl.plot(x, var_fitted, 'r', linewidth=2, label='model')
            pl.ylabel(obj)
            pl.legend()
            pl.savefig(save_dir+'/bestind.png')
            pl.show()

        error[ind][-1] = np.sum(error[ind])

    print 'Error of best individual: '
    for i, obj in enumerate(optimizer_json['objectives']):
        print obj + ': ' + str(error[0][i])

    with open(save_dir + '/error.json', 'r') as file:
        error_saved = json_utils.load(file)
    print 'Saved error of best individual: '
    for obj in optimizer_json['objectives']:
        print obj + ': ' + str(error_saved[obj][-1])

    pl.figure()
    pl.plot(range(len(error_saved[obj])), error_saved[obj], 'k', linewidth=2)
    pl.ylabel('Error')
    pl.xlabel('Generation')
    pl.savefig(save_dir+'/error.png')
    pl.show()
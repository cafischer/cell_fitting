from __future__ import division
import numpy as np
from neuron import h
from model.cell_builder import *
import json_utils
from optimizer import mean_squared_error, extract_simulation_params
import optimization
import optimization.fitfuns
h.load_file("stdrun.hoc")  # load NEURON libraries
h("""cvode.active(0)""")  # invariable time step in NEURON

from optimize_active.optimize_activepointaxon import *

__author__ = 'caro'


def make_optimizer_json(objective, data_dir, mechanism_dir, simulation_params, fun_to_fit, var_to_fit,
                        recalculate_variables=None):
    optimizer_json = dict()
    optimizer_json['objectives'] = [objective]
    optimizer_json['data_dir'] = {objective: data_dir}
    optimizer_json['mechanism_dir'] = mechanism_dir
    data = dict()
    data[objective] = pd.read_csv(os.path.join(os.path.abspath('..'), optimizer_json['data_dir'][objective]))
    optimizer_json['simulation_params'] = {objective: extract_simulation_params(data)}
    optimizer_json['simulation_params'][objective].update(simulation_params)
    optimizer_json['fun_to_fit'] = {objective: fun_to_fit}
    optimizer_json['var_to_fit'] = {objective: var_to_fit}
    optimizer_json['recalculate_variables'] = recalculate_variables
    return optimizer_json


if __name__ == "__main__":

    cellid = '2015_08_11d'
    save_dir = './optimize_active/results_active/new_cells/' + cellid
    individuals = ['0']
    useobj = False
    objective = 'stepcurrent-0.1'

    # load the jsonized optimizer
    if useobj:
        optimizer_json = make_optimizer_json(objective=objective,
                                            data_dir='data/new_cells/'+cellid+'/stepcurrent/'+objective+'.csv',
                                            mechanism_dir='model/channels',
                                            simulation_params={'onset': {objective: 200}},
                                            fun_to_fit='run_simulation', var_to_fit='v')
    else:
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
    cell = Cell(save_dir + '/cell.json', optimizer_json['mechanism_dir'])  # saver when models are changed

    # load data
    data = dict()
    for obj in optimizer_json['objectives']:
        data[obj] = pd.read_csv(os.path.join(os.path.abspath('..'), optimizer_json['data_dir'][obj]))

    # look at individuals
    error = np.zeros(len(individuals), dtype=object) # save error of each individual for all objectives (for comparison)
    for i, ind in enumerate(individuals):
        error[ind] = [0] * (len(optimizer_json['objectives'])+1)

        for j, obj in enumerate(optimizer_json['objectives']):

            # update the cell with variables from the best individuals
            with open(save_dir + '/' + 'variables_new_'+str(ind)+'.json', 'r') as file: #TODO
            #with open(save_dir + '/' + 'variables_new_'+obj+'.json', 'r') as file:
                variables_new = json.load(file)
            for k, p in enumerate(variables_new):
                for path in variables_new[k][2]:
                    cell.update_attr(path, variables_new[k][1])
            if optimizer_json['recalculate_variables'] is not None:
                variables_new_dict = dict()
                for var in variables_new:
                    variables_new_dict[var[0]] = var[1]
                recalculate_variables = getattr(optimization.fitfuns, optimizer_json['recalculate_variables'])
                recalculate_variables(variables_new_dict)

            # run simulation and compute the variable to fit
            fun_to_fit = getattr(optimization.fitfuns,
                                 str(optimizer_json['fun_to_fit'][obj]))
            var_fitted, x = fun_to_fit(cell, **optimizer_json['simulation_params'][obj])

            # data to fit
            data_to_fit = np.array(data[obj][optimizer_json['var_to_fit'][obj]])  # convert to array
            data_to_fit = data_to_fit[~np.isnan(data_to_fit)]  # get rid of nans

            # compute error
            error[i][j] = mean_squared_error(var_fitted, data_to_fit)

            # plot the results
            pl.figure()
            pl.plot(x, data_to_fit, 'k', linewidth=2, label='data')
            pl.plot(x, var_fitted, 'r', linewidth=2, label='model')
            #pl.ylabel(obj)
            pl.xlabel('Time (ms)', fontsize=18)
            pl.ylabel('Membrane \npotential (mV)', fontsize=18)
            pl.legend(loc='lower right', fontsize=18)
            if ind == '0': pl.savefig(save_dir+'/bestind_'+obj+'.png')
            pl.show()

        error[i][-1] = np.sum(error[ind])

    # show error development
    print 'Error of best individual: '
    for i, obj in enumerate(optimizer_json['objectives']):
        print obj + ': ' + str(error[0][i])
    np.savetxt(save_dir+'/error.txt', np.array(error[0]))

    if not useobj:
        with open(save_dir + '/error.json', 'r') as file:
            error_saved = json_utils.load(file)
        print 'Saved error of best individual (per objective): '
        for obj in optimizer_json['objectives']:
            print obj + ': ' + str(error_saved[obj][-1])

            pl.figure()
            pl.plot(range(len(error_saved[obj])), error_saved[obj], 'k', linewidth=2)
            pl.ylabel('Error')
            pl.xlabel('Generation')
            pl.savefig(save_dir+'/error_'+obj+'.png')
            pl.show()
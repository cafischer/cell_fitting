from inspyred import ec
import numpy as np
import pandas as pd
from optimization import fitfuns
from optimization.optimizer import extract_simulation_params
from model.cell_builder import complete_mechanismdir, Cell
from neuron import h
from optimization.bioinspired.performance_test import errfuns

__author__ = 'caro'


class Problem:

    def __init__(self, params):
        self.lower_bound = params['lower_bound']
        self.upper_bound = params['upper_bound']
        self.maximize = params['maximize']
        self.model_dir = params['model_dir']
        self.recalculate_variables = None
        self.fun_to_fit = getattr(fitfuns, params['fun_to_fit'])
        self.var_to_fit = params['var_to_fit']
        self.data = pd.read_csv(params['data_dir'])
        self.simulation_params = extract_simulation_params(self.data)
        h.nrn_load_dll(complete_mechanismdir(params['mechanism_dir']))
        self.path_variables = params['path_variables']
        self.errfun = getattr(errfuns, params['errfun'])

    def generator(self, random, args):
        n_variables = len(self.path_variables)
        if np.size(self.lower_bound) == 1 and np.size(self.upper_bound) == 1:
            return np.array([random.uniform(self.lower_bound, self.upper_bound) for i in range(n_variables)])
        elif np.size(self.lower_bound) == n_variables and np.size(self.upper_bound) == n_variables:
            return np.array([random.uniform(self.lower_bound[i], self.upper_bound[i]) for i in range(n_variables)])
        else:
            raise ValueError('Size of upper or lower boundary is unequal to 1 or the number of variables!')

    def bounder(self, lower_bound, upper_bound):
        return ec.Bounder(lower_bound, upper_bound)

    def evaluator(self, candidates, args):
        fitness = []
        for candidate in candidates:
            fitness.append(self.evaluate(candidate, args))
        return fitness

    def evaluate(self, candidate, args):
        # TODO
        #candidate = [0.05, 0. [0.06, 0.07]

        var_to_fit = self.get_var_to_fit(candidate)

        # compute fitness
        data_to_fit = np.array(self.data[self.var_to_fit])  # convert to array
        fitness = self.errfun(var_to_fit, data_to_fit, np.array(self.data.t))

        # TODO: save one model
        #import csv
        #pl.figure()
        #pl.plot(np.array(self.data.t), var_to_fit)
        #pl.show()
        #data_new = np.column_stack((np.array(self.data.t), np.array(self.data.i), var_to_fit,
        #                            np.zeros(len(var_to_fit), dtype=object)))
        #data_new[0, 3] = 'soma'
        #with open('./testdata/modeldata.csv', 'w') as csvoutput:
        #    writer = csv.writer(csvoutput, lineterminator='\n')
        #    writer.writerow(['t', 'i', 'v', 'sec'])
        #    writer.writerows(data_new)
        return fitness

    def get_var_to_fit(self, candidate):
        # create cell with the candidate variables
        cell = Cell.from_modeldir(self.model_dir)
        for i in range(len(candidate)):
            for path in self.path_variables[i]:
                cell.update_attr(path, candidate[i])
        if self.recalculate_variables is not None:
            self.recalculate_variables(candidate)

        # run simulation and compute the variable to fit
        var_to_fit, _ = self.fun_to_fit(cell, **self.simulation_params)
        return var_to_fit



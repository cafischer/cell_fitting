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
        if params['mechanism_dir'] is not None:
            h.nrn_load_dll(str(complete_mechanismdir(params['mechanism_dir'])))
        self.path_variables = params['path_variables']
        self.errfun = getattr(errfuns, params['errfun'])

    def generator(self, random, args):
        n_variables = len(self.path_variables)

        # normalize variables
        if args['normalize']:
            lower_bound = 0
            upper_bound = 1
        else:
            lower_bound = self.lower_bound
            upper_bound = self.upper_bound

        if np.size(lower_bound) == 1 and np.size(upper_bound) == 1:
            return np.array([random.uniform(lower_bound, upper_bound) for i in range(n_variables)])
        elif np.size(lower_bound) == n_variables and np.size(upper_bound) == n_variables:
            return np.array([random.uniform(lower_bound[i], upper_bound[i]) for i in range(n_variables)])
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

        # unnormalize
        if args['normalize']:
            candidate_unnormed = [candidate[i] * (self.upper_bound-self.lower_bound) + self.lower_bound for i in range(len(candidate))]
        else:
            candidate_unnormed = candidate

        cell = self.get_cell(candidate_unnormed)

        # run simulation and compute the variable to fit
        var_to_fit, _ = self.fun_to_fit(cell, **self.simulation_params)

        # compute fitness
        data_to_fit = np.array(self.data[self.var_to_fit])  # convert to array
        fitness = self.errfun(var_to_fit, data_to_fit, np.array(self.data.t))
        return fitness

    def get_cell(self, candidate):
        # create cell with the candidate variables
        cell = Cell.from_modeldir(self.model_dir)
        for i in range(len(candidate)):
            for path in self.path_variables[i]:
                cell.update_attr(path, candidate[i])
        if self.recalculate_variables is not None:
            self.recalculate_variables(candidate)
        return cell

    def observer(self, population, num_generations, num_evaluations, args):
        """
        Stores the current generation in args['individuals_file'] with columns ['generation', 'number', 'fitness',
        'candidate'].
        Generation: Generation of the candidate
        Number: Index of the candidate in this generation
        Fitness: Fitness of the candidate
        Candidate: String representation of the current candidate
        :param population: Actual population.
        :type population: list
        :param num_generations: Actual generation.
        :type num_generations: int
        :param num_evaluations: Actual number of evaluations.
        :type num_evaluations: int
        :param args: Additional arguments. Should contain a file object under the keyword 'individuals_file'
        :type args: dict
        """

        pop_size = len(population)
        fitness = np.array([population[j].fitness for j in range(pop_size)])
        candidates = np.array([population[j].candidate for j in range(pop_size)])
        # unnormalize
        if args['normalize']:
            for candidate in candidates:
                for j in range(len(candidate)):
                    candidate[j] = candidate[j] * (self.upper_bound-self.lower_bound) + self.lower_bound
        # convert candidates to string
        candidates = np.array([str(candidate) for candidate in candidates])
        idx = np.argsort(fitness)

        # create DataFrame for this generation
        individuals_file = pd.DataFrame(index=(num_generations*pop_size)+np.array(range(pop_size)),
                                            columns=['generation', 'number', 'fitness', 'candidate'])
        individuals_file.generation = num_generations
        individuals_file.number = range(pop_size)
        individuals_file.fitness = fitness[idx]
        individuals_file.candidate = candidates[idx]

        # save DataFrame from this generation
        if num_generations == 0:
            individuals_file.to_csv(args['individuals_file'], header=True)
        else:
            individuals_file.to_csv(args['individuals_file'], header=False)


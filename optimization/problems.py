from abc import ABCMeta, abstractmethod

from inspyred import ec
import numpy as np
import pandas as pd
from neuron import h
from nrn_wrapper import Cell, complete_mechanismdir

from optimization.simulate import run_simulation, extract_simulation_params
from optimization import errfuns, fitfuns
from optimization.bio_inspired.inspyred_extension.observers import individuals_observer
from optimization.bio_inspired.inspyred_extension.generators import bybounds_generator

__author__ = 'caro'


def unnorm_candidate(candidate, lower_bound, upper_bound):

    return candidate * (upper_bound-lower_bound) + lower_bound


def norm_candidate(candidate, lower_bound, upper_bound):

    return (candidate - lower_bound) / (upper_bound-lower_bound)


def unnorm_population(population, lower_bound, upper_bound):

    population_unnormed = np.zeros(len(population), dtype=object)
    for i, p in enumerate(population):
        population_unnormed[i] = ec.Individual()  # explicitly copy because candidate setter changes fitness to None
        population_unnormed[i].candidate = p.candidate * (upper_bound-lower_bound) + lower_bound
        population_unnormed[i].fitness = p.fitness
        population_unnormed[i].birthdate = p.birthdate
        population_unnormed[i].maximize = p.maximize
    return population_unnormed


def insert_mechanisms(cell, path_variables):
    try:
        for paths in path_variables:
            for path in paths:
                cell.get_attr(path[:-3]).insert(path[-2])  # [-3]: pos (not needed insert into section)
                                                           # [-2]: mechanism, [-1]: attribute
    except AttributeError:
        pass  # let all non mechanism variables pass


def get_ionlist(channel_list):
    """
    Get the ion names by the convention that the ion name is included in the channel name.
    :param channel_list: List of channel names.
    :type channel_list: list
    :return: List of ion names.
    :rtype: list
    """
    ion_list = []
    for channel in channel_list:
        if 'na' in channel:
            ion_list.append('na')
        elif 'k' in channel:
            ion_list.append('k')
        elif 'ca' in channel:
            ion_list.append('ca')
        else:
            ion_list.append('')
    return ion_list


def convert_units(L, diam, cm, dvdt, i_inj, currents):
    cell_area = L * diam * np.pi * 1e-8  # cm
    Cm = cm * cell_area  # uF

    i_inj_sc = i_inj * 1e-9  # A
    dvdt_sc = dvdt * 1e-6  # A
    currents_sc = []
    for i in range(len(currents)):
        currents_sc.append(currents[i] * cell_area * 1e-3)  # A
    return dvdt_sc, i_inj_sc, currents_sc, Cm, cell_area

class Problem(object):
    __metaclass__ = ABCMeta

    def __init__(self, maximize):
        self.maximize = maximize

    @abstractmethod
    def generator(self, random, args):
        pass

    @abstractmethod
    def evaluator(self, candidates, args):
        pass

    @abstractmethod
    def bounder(self, candidate, args):
        pass


class CellFitProblem(Problem):

    def __init__(self, maximize, normalize, model_dir, mechanism_dir, variables, data_dir, get_var_to_fit,
                 fitnessweights, errfun, insert_mechanisms=False, recalculate_variables=None):
        super(CellFitProblem, self).__init__(maximize)
        self.normalize = normalize
        self.model_dir = model_dir
        if mechanism_dir is not None:
            h.nrn_load_dll(str(complete_mechanismdir(mechanism_dir)))

        self.lower_bound = np.zeros(len(variables))
        self.upper_bound = np.zeros(len(variables))
        self.path_variables = list()
        for i, var in enumerate(variables):
            self.lower_bound[i] = var[0]
            self.upper_bound[i] = var[1]
            self.path_variables.append(var[2])

        self.fitnessweights = np.array(fitnessweights)
        self.errfun = getattr(errfuns, errfun)
        self.recalculate_variables = recalculate_variables
        self.insert_mechanisms = insert_mechanisms

        # read in data, extract simulation parameters, get variable to fit and data to fit
        if np.size(data_dir) == 1:
            self.data = pd.read_csv(data_dir)
            self.simulation_params = extract_simulation_params(self.data)
            self.fitfuns = getattr(fitfuns, get_var_to_fit)
            self.data_to_fit = self.fitfuns(self.data.v.values, self.data.t.values, self.data.i.values)
        else:
            self.data = list()
            self.simulation_params = list()
            self.fitfuns = list()
            self.data_to_fit = list()
            for i in range(len(data_dir)):
                self.data.append(pd.read_csv(data_dir[i]))
                self.simulation_params.append(extract_simulation_params(self.data[i]))
                self.fitfuns.append(getattr(fitfuns, get_var_to_fit[i]))
                self.data_to_fit.extend(self.fitfuns[i](self.data[i].v.values, self.data[i].t.values,
                                                        self.data[i].i.values))


    def generator(self, random, args):
        n_vars = len(self.path_variables)

        # normalize variables
        if self.normalize:
            return bybounds_generator(random, n_vars, 0, 1)

        return bybounds_generator(random, n_vars, self.lower_bound, self.upper_bound)

    def bounder(self, candidate, args):
        if self.normalize:
            return ec.Bounder(0, 1)(candidate, args)

        return ec.Bounder(self.lower_bound, self.upper_bound)(candidate, args)

    def evaluator(self, candidates, args):
        fitness = list()
        for candidate in candidates:
            fitness.append(self.evaluate(candidate, args))
        return fitness

    def evaluate(self, candidate, args):

        if self.normalize:
            candidate = unnorm_candidate(candidate, self.lower_bound, self.upper_bound)

        # create cell
        cell = self.get_cell(candidate)

        vars_to_fit = list()
        if np.size(self.simulation_params) == 1:
            # run simulation
            v_candidate, t_candidate = run_simulation(cell, **self.simulation_params)

            # compute the variables to fit
            vars_to_fit.extend(self.fitfuns(v_candidate, t_candidate, self.simulation_params['i_amp']))
        else:  # in case you want to run several simulations on different data
            for i in range(len(self.simulation_params)):
                # run simulation
                v_candidate, t_candidate = run_simulation(cell, **self.simulation_params[i])

                # compute the variables to fit
                vars_to_fit.extend(self.fitfuns[i](v_candidate, t_candidate, self.simulation_params[i]['i_amp']))

        # compute fitness
        if None in vars_to_fit:
            return np.inf

        fitness = np.array([self.errfun(vars_to_fit[i], self.data_to_fit[i]) for i in range(len(self.data_to_fit))])

        # check for Nones, weighting and sum fitness
        fitness = np.sum(self.fitnessweights * fitness)
        return fitness

    def get_cell(self, candidate):
        # create cell with the candidate variables
        cell = Cell.from_modeldir(self.model_dir)

        if self.insert_mechanisms:
            insert_mechanisms(cell, self.path_variables)

        for i in range(len(candidate)):
            for path in self.path_variables[i]:
                cell.update_attr(path, candidate[i])
        if self.recalculate_variables is not None:
            self.recalculate_variables(candidate)

        return cell

    def observer(self, population, num_generations, num_evaluations, args):
        """
        Stores the current generation in args['individuals_file'] with columns ['generation', 'number', 'fitness',
        'candidate']. Unnorms the candidate if necessary
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
        if self.normalize:
            population_unnormed = unnorm_population(population, self.lower_bound, self.upper_bound)
            return individuals_observer(population_unnormed, num_generations, num_evaluations, args)

        return individuals_observer(population, num_generations, num_evaluations, args)


class CellFitFromInitPopProblem(CellFitProblem):

    def __init__(self, maximize, normalize, model_dir, mechanism_dir, variables, data_dir, get_var_to_fit,
                 fitnessweights, errfun, init_candidate, insert_mechanisms=False, recalculate_variables=None):
        super(CellFitFromInitPopProblem, self).__init__(maximize, normalize, model_dir, mechanism_dir, variables,
                                                        data_dir, get_var_to_fit, fitnessweights, errfun,
                                                        insert_mechanisms, recalculate_variables)
        self.init_candidate = init_candidate

    def generator(self, random, args):
        # initial population will consist of the same candidate
        candidate = self.init_candidate
        if self.normalize:
            candidate = norm_candidate(candidate, self.lower_bound, self.upper_bound)
        return np.array(candidate)
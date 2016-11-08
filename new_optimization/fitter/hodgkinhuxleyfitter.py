import pandas as pd
import numpy as np
from nrn_wrapper import Cell, load_mechanism_dir, iclamp
from optimization import errfuns
from optimization import fitfuns
from optimization.simulate import extract_simulation_params

__author__ = 'caro'


def insert_mechanisms(cell, path_variables):  # TODO: think of better method to insert/identify mechanisms
    try:
        for paths in path_variables:
            for path in paths:
                cell.get_attr(path[:-3]).insert(path[-2])  # [-3]: pos (not needed insert into section)
                                                           # [-2]: mechanism, [-1]: attribute
    except AttributeError:
        pass  # let all non mechanism variables pass


class HodgkinHuxleyFitter:

    def __init__(self, variable_keys, errfun_name, fitfun_names, fitnessweights,
                 model_dir, mechanism_dir, data_dir, simulation_params=None, args=None):

        self.variable_keys = variable_keys
        self.errfun_names = errfun_name
        self.fitfun_names = fitfun_names
        self.errfun = getattr(errfuns, errfun_name)
        self.fitfuns = [getattr(fitfuns, fitfun_name) for fitfun_name in fitfun_names]
        self.fitnessweights = fitnessweights
        self.args = args
        self.model_dir = model_dir
        self.mechanism_dir = mechanism_dir
        if mechanism_dir is not None:
            load_mechanism_dir(mechanism_dir)
        self.data_dir = data_dir
        self.data = pd.read_csv(data_dir)
        self.init_simulation_params = simulation_params
        if simulation_params is None:
            simulation_params = {}
        self.simulation_params = extract_simulation_params(self.data, **simulation_params)
        self.data_to_fit = [fitfun(self.data.v.values, self.data.t.values, self.data.i.values, self.args)
                            for fitfun in self.fitfuns]
        self.cell = self.get_cell()

    def evaluate_fitness(self, candidate, args):
        self.update_cell(candidate)
        v_candidate, t_candidate = iclamp(self.cell, **self.simulation_params)
        #self.data_to_fit = [0]  # TODO !!!!
        #self.args['candidate'] = candidate  # TODO
        vars_to_fit = [fitfun(v_candidate, t_candidate, self.simulation_params['i_inj'], self.args)
                       for fitfun in self.fitfuns]
        num_nones = 0
        fitness = 0
        for i in range(len(vars_to_fit)):
            if vars_to_fit[i] is None:
                num_nones += 1
            else:
                fitness += self.fitnessweights[i] * self.errfun(vars_to_fit[i], self.data_to_fit[i])
        if num_nones == len(vars_to_fit):
            return float("inf")
        return fitness

    def get_cell(self):
        cell = Cell.from_modeldir(self.model_dir)
        insert_mechanisms(cell, self.variable_keys)
        return cell

    def update_cell(self, candidate):
        for i in range(len(candidate)):
            for path in self.variable_keys[i]:
                self.cell.update_attr(path, candidate[i])

    def simulate_cell(self, candidate):
        self.update_cell(candidate)
        v_candidate, t_candidate = iclamp(self.cell, **self.simulation_params)
        return v_candidate, t_candidate, self.simulation_params['i_inj']

    def to_dict(self):
        return {'variable_keys': self.variable_keys, 'errfun': self.errfun_names, 'fitfun': self.fitfun_names,
                'fitnessweights': self.fitnessweights, 'model_dir': self.model_dir, 'mechanism_dir': self.mechanism_dir,
                'data_dir': self.data_dir, 'simulation_params': self.init_simulation_params}  #, 'args': self.args}  # TODO
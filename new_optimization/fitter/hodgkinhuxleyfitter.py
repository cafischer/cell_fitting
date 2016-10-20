import pandas as pd
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

    def __init__(self, variable_keys, errfun, fitfun, fitnessweights,
                 model_dir, mechanism_dir, data_dir, simulation_params=None, args=None):

        self.variable_keys = variable_keys
        self.errfun_name = errfun
        self.fitfun_name = fitfun
        self.errfun = getattr(errfuns, errfun)
        self.fitfun = getattr(fitfuns, fitfun)
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
        self.data_to_fit = self.fitfun(self.data.v.values, self.data.t.values, self.data.i.values, self.args)
        self.cell = self.get_cell()

    def evaluate_fitness(self, candidate, args):
        self.update_cell(candidate)
        v_candidate, t_candidate = iclamp(self.cell, **self.simulation_params)
        vars_to_fit = self.fitfun(v_candidate, t_candidate, self.simulation_params['i_inj'], self.args)
        if any(var is None for var in vars_to_fit):
            return float("inf")

        fitness = [self.errfun(vars_to_fit[i], self.data_to_fit[i]) for i in range(len(self.data_to_fit))]
        fitness_weighted = 0
        for i in range(len(fitness)):
            fitness_weighted += self.fitnessweights[i] * fitness[i]
        return fitness_weighted

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
        return {'variable_keys': self.variable_keys, 'errfun': self.errfun_name, 'fitfun': self.fitfun_name,
                'fitnessweights': self.fitnessweights, 'model_dir': self.model_dir, 'mechanism_dir': self.mechanism_dir,
                'data_dir': self.data_dir, 'simulation_params': self.init_simulation_params, 'args': self.args}
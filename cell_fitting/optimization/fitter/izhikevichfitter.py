import pandas as pd
from cell_fitting.izhikevichStellateCell import get_v_izhikevich, get_v_izhikevich_vector2d

__author__ = 'caro'


class IzhikevichFitter:

    all_variable_names = ['v_rest', 'v_reset', 'v_peak', 'cm', 'k_rest', 'k_t', 'v_t', 'a1', 'a2', 'b1', 'b2',
                          'd1', 'd2', 'i_b', 'v0', 'u0']

    def __init__(self, name, variable_keys, given_variables, fitfuns, errfun, data_dirs, data_to_fit=None):
        self.name = name
        self.variable_keys = variable_keys
        self.name_value_variables = dict()
        for name in IzhikevichFitter.all_variable_names:
            self.name_value_variables[name] = given_variables.get(name)
        self.fitfuns = fitfuns
        self.errfun = errfun
        self.readin_data(data_dirs)
        self.extract_simulation_params()
        self.set_data_to_fit(data_to_fit)

    def readin_data(self, data_dirs):
        self.data = [pd.read_csv(dir) for dir in data_dirs]

    def extract_simulation_params(self):
        self.simulation_params = [{'tstop': data_set.t.values[-1], 'dt': data_set.t.values[1],
                                   'i_inj': data_set.i.values} for data_set in self.data]

    def set_data_to_fit(self, data_to_fit):
        if data_to_fit is None:
            self.data_to_fit = [self.fitfuns[i](data_set.v.values, data_set.t.values, data_set.i.values)
                                for i, data_set in enumerate(self.data)]
        else:
            self.data_to_fit = data_to_fit

    def evaluate(self, candidate, args):
        fitness = 0

        for name, value in zip(self.variable_keys, candidate):
            self.name_value_variables[name] = value

        for i, simulation_params in enumerate(self.simulation_params):
            v_candidate, t_candidate, _ = get_v_izhikevich_vector2d(simulation_params['i_inj'],
                                                                    simulation_params['tstop'],
                                                                    simulation_params['dt'],
                                                                    **self.name_value_variables)
            vars_to_fit = self.fitfuns[i](v_candidate, t_candidate, simulation_params['i_inj'])
            if vars_to_fit is None:
                return float("inf")
            fitness += self.errfun(vars_to_fit, self.data_to_fit[i])
        return fitness


class IzhikevichFitter1d:

    all_variable_names = ['v_rest', 'v_reset', 'v_peak', 'cm', 'k_rest', 'k_t', 'v_t', 'a', 'b',
                          'd', 'i_b', 'v0', 'u0']

    def __init__(self, name, variable_keys, given_variables, fitfuns, errfun, data_dirs, data_to_fit=None):
        self.name = name
        self.variable_keys = variable_keys
        self.name_value_variables = dict()
        for name in IzhikevichFitter1d.all_variable_names:
            self.name_value_variables[name] = given_variables.get(name)
        self.fitfuns = fitfuns
        self.errfun = errfun
        self.readin_data(data_dirs)
        self.extract_simulation_params()
        self.set_data_to_fit(data_to_fit)

    def readin_data(self, data_dirs):
        self.data = [pd.read_csv(dir) for dir in data_dirs]

    def extract_simulation_params(self):
        self.simulation_params = [{'tstop': data_set.t.values[-1], 'dt': data_set.t.values[1],
                                   'i_inj': data_set.i.values} for data_set in self.data]

    def set_data_to_fit(self, data_to_fit):
        if data_to_fit is None:
            self.data_to_fit = [self.fitfuns[i](data_set.v.values, data_set.t.values, data_set.i.values)
                                for i, data_set in enumerate(self.data)]
        else:
            self.data_to_fit = data_to_fit

    def evaluate(self, candidate, args):
        fitness = 0

        for name, value in zip(self.variable_keys, candidate):
            self.name_value_variables[name] = value

        for i, simulation_params in enumerate(self.simulation_params):
            v_candidate, t_candidate, _ = get_v_izhikevich(simulation_params['i_inj'],
                                                                simulation_params['tstop'],
                                                                simulation_params['dt'],
                                                                **self.name_value_variables)
            vars_to_fit = self.fitfuns[i](v_candidate, t_candidate, simulation_params['i_inj'])
            if vars_to_fit is None:
                return float("inf")
            fitness += self.errfun(vars_to_fit, self.data_to_fit[i])
        return fitness
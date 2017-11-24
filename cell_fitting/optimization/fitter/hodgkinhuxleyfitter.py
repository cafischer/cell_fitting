from __future__ import division
import numpy as np
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization import errfuns, fitfuns
from cell_fitting.optimization.fitter.fitter_interface import Fitter
from cell_fitting.optimization.fitter.read_data import read_data
from cell_fitting.optimization.simulate import iclamp_handling_onset, iclamp_adaptive_handling_onset, \
    extract_simulation_params
from cell_fitting.util import merge_dicts, init_nan

__author__ = 'caro'


class HodgkinHuxleyFitter(Fitter):

    def __init__(self, name, variable_keys, errfun_name, fitfun_names_per_data_set, fitnessweights_per_data_set,
                 data_read_dict_per_data_set, model_dir, mechanism_dir, init_simulation_params=None, args=None):
        super(HodgkinHuxleyFitter, self).__init__()
        self.name = name
        self.variable_keys = variable_keys
        self.args = args

        self.errfun_names = errfun_name
        self.errfun = getattr(errfuns, errfun_name)

        self.fitfun_names_per_data_set = fitfun_names_per_data_set
        self.fitfuns = [[getattr(fitfuns, fitfun_name) for fitfun_name in fitfun_names]
                        for fitfun_names in fitfun_names_per_data_set]
        self.fitnessweights_per_data_set = fitnessweights_per_data_set

        self.data_read_dict_per_data_set = data_read_dict_per_data_set
        data_dicts = list()
        for data_read_dict in self.data_read_dict_per_data_set:
            data_dicts.append(read_data(**data_read_dict))

        self.init_simulation_params = init_simulation_params if not None else {}
        if type(self.init_simulation_params) is dict:
            self.init_simulation_params = [self.init_simulation_params] * len(data_dicts)

        self.simulation_params = []
        self.data_sets_to_fit = []
        for i, data_dict in enumerate(data_dicts):
            extracted_params = extract_simulation_params(**data_dict)
            self.simulation_params.append(merge_dicts(extracted_params, self.init_simulation_params[i]))
            self.data_sets_to_fit.append([fitfun(args=self.args, **data_dict) for fitfun in self.fitfuns[i]])

        self.model_dir = model_dir
        self.mechanism_dir = mechanism_dir
        if mechanism_dir is not None:
            load_mechanism_dir(mechanism_dir)
        self.cell = self.get_cell()

    def get_cell(self):
        cell = Cell.from_modeldir(self.model_dir)
        cell.insert_mechanisms(self.variable_keys)
        return cell

    def update_cell(self, candidate):
        for i in range(len(candidate)):
            for path in self.variable_keys[i]:
                self.cell.update_attr(path, candidate[i])

    def simulate_cell(self, candidate, simulation_params):
        self.update_cell(candidate)
        v_candidate, t_candidate, i_inj = iclamp_handling_onset(self.cell, **simulation_params)
        return v_candidate, t_candidate, i_inj

    def evaluate_fitness(self, candidate, args):
        fitness = 0
        max_fitness_error = self.args.pop('max_fitness_error', np.inf)

        for s, simulation_params in enumerate(self.simulation_params):
            v_candidate, t_candidate, _ = self.simulate_cell(candidate, simulation_params)
            vars_to_fit = [fitfun(v_candidate, t_candidate, simulation_params['i_inj'], self.args)
                           for fitfun in self.fitfuns[s]]

            for i in range(len(vars_to_fit)):
                if vars_to_fit[i] is None:
                    fitness = max_fitness_error
                    break
                else:
                    fitness += self.fitnessweights_per_data_set[s][i] * self.errfun(vars_to_fit[i], self.data_sets_to_fit[s][i])
            v_candidate, t_candidate, _ = self.simulate_cell(candidate, simulation_params)  # TODO
        if np.isnan(fitness):
            return max_fitness_error
        return fitness

    def to_dict(self):
        return {'name': self.name, 'variable_keys': self.variable_keys,
                'errfun_name': self.errfun_names,
                'fitfun_names_per_data_set': self.fitfun_names_per_data_set,
                'fitnessweights_per_data_set': self.fitnessweights_per_data_set,
                'data_read_dict_per_data_set': self.data_read_dict_per_data_set,
                'model_dir': self.model_dir, 'mechanism_dir': self.mechanism_dir,
                'init_simulation_params': self.init_simulation_params,
                'args': self.args}



class HodgkinHuxleyFitterAdaptive(HodgkinHuxleyFitter):

    def __init__(self, name, variable_keys, errfun_name, fitfun_names_per_data_set, fitnessweights_per_data_set,
                 data_read_dict_per_data_set, model_dir, mechanism_dir, init_simulation_params=None, args=None):
        super(HodgkinHuxleyFitterAdaptive, self).__init__(name, variable_keys, errfun_name, fitfun_names_per_data_set,
                                                          fitnessweights_per_data_set, data_read_dict_per_data_set,
                                                          model_dir, mechanism_dir, init_simulation_params,
                                                          args)

        self.data_read_dict_per_data_set = data_read_dict_per_data_set
        data_dicts = []
        for i, data_read_dict in enumerate(self.data_read_dict_per_data_set):
            data_dict, discontinuities = read_data(return_discontinuities=True, **data_read_dict)
            data_dicts.append(data_dict)
            self.init_simulation_params = merge_dicts(self.init_simulation_params[i],
                                                      {'discontinuities': discontinuities,
                                                       'interpolate': True, 'continuous': True})

    def simulate_cell(self, candidate, simulation_params):
        self.update_cell(candidate)
        try:
            v_candidate, t_candidate, i_inj = iclamp_adaptive_handling_onset(self.cell, **simulation_params)
        except RuntimeError:  # prevents failing when adaptive integration cannot find solution
            v_candidate = t_candidate = i_inj = init_nan(len(simulation_params['i_inj']))
        return v_candidate, t_candidate, i_inj


# TODO: all below
# class HodgkinHuxleyFitterPareto(HodgkinHuxleyFitter):
#
#     def __init__(self, name, variable_keys, errfun_name, fitfun_names, fitnessweights,
#                  model_dir, mechanism_dir, data_dir, simulation_params=None, args=None):
#         super(HodgkinHuxleyFitterPareto, self).__init__(name, variable_keys, errfun_name, fitfun_names,
#                                                                 fitnessweights, model_dir, mechanism_dir, data_dir,
#                                                                 simulation_params, args)
#
#     def evaluate_fitness(self, candidate, args):
#         fitnesses = list()
#         for i, fitfun in enumerate(self.fitfuns):
#             fitnesses.append(self.evaluate_fitfun(candidate, fitfun, self.fitnessweights_per_data_set[i], self.data_to_fit[i]))
#         return Pareto(fitnesses)
#
#     def evaluate_fitfun(self, candidate, fitfun, fitnessweight, data_to_fit):
#         v_candidate, t_candidate, _ = self.simulate_cell(candidate)
#         var_to_fit = fitfun(v_candidate, t_candidate, self.simulation_params['i_inj'], self.args)
#         if var_to_fit is None:
#             return 1000 #float("inf") TODO
#         fitness = fitnessweight * self.errfun(var_to_fit, data_to_fit)
#         return fitness
#
#
# class HodgkinHuxleyFitterCurrentPenalty(HodgkinHuxleyFitter):
#
#     def __init__(self, name, variable_keys, errfun_name, fitfun_names, fitnessweights,
#                      model_dir, mechanism_dir, data_dir, simulation_params=None, args=None):
#         super(HodgkinHuxleyFitterCurrentPenalty, self).__init__(name, variable_keys, errfun_name, fitfun_names,
#                                                                 fitnessweights, model_dir, mechanism_dir, data_dir,
#                                                                 simulation_params, args)
#
#     def evaluate_fitness(self, candidate, args):
#
#         v_candidate, t_candidate, currents = self.simulate_cell_with_currents(candidate)
#         vars_to_fit = [fitfun(v_candidate, t_candidate, self.simulation_params['i_inj'], self.args)
#                        for fitfun in self.fitfuns]
#         fitness = 0
#         for i in range(len(vars_to_fit)):
#             if vars_to_fit[i] is None:
#                 fitness = 10000
#                 break
#             else:
#                 fitness += self.fitnessweights_per_data_set[i] * self.errfun(vars_to_fit[i], self.data_to_fit[i])
#         current_penalty = np.sum(np.sum(np.abs(currents))) / np.size(currents)
#         fitness += current_penalty
#         return fitness
#
#     def simulate_cell_with_currents(self, candidate):
#         channel_list = get_channel_list(self.cell, 'soma')
#         ion_list = get_ionlist(channel_list)
#
#         # record currents
#         currents = np.zeros(len(channel_list), dtype=object)
#         for i in range(len(channel_list)):
#             currents[i] = self.cell.soma.record_from(channel_list[i], 'i' + ion_list[i], pos=.5)
#
#         v_candidate, t_candidate, _ = self.simulate_cell(candidate)
#         currents = [np.array(c) for c in currents]
#         return v_candidate, t_candidate, currents
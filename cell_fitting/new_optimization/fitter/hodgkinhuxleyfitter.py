from __future__ import division
import pandas as pd
import numpy as np
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization import errfuns
from cell_fitting.optimization import fitfuns
from cell_fitting.optimization.simulate import iclamp_handling_onset, iclamp_adaptive_handling_onset, extract_simulation_params
from cell_fitting.util import merge_dicts
import functools
from inspyred.ec.emo import Pareto
from cell_fitting.new_optimization.fitter.fitter_interface import Fitter
from cell_fitting.optimization.helpers import get_channel_list, get_ionlist

__author__ = 'caro'


class HodgkinHuxleyFitter(Fitter):

    def __init__(self, name, variable_keys, errfun_name, fitfun_names, fitnessweights,
                 model_dir, mechanism_dir, data_dir, simulation_params=None, args=None):
        super(HodgkinHuxleyFitter, self).__init__()
        self.name = name
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
        self.init_simulation_params = simulation_params if not None else {}
        self.simulation_params = merge_dicts(extract_simulation_params(self.data), self.init_simulation_params)
        self.data_to_fit = [fitfun(self.data.v.values, self.data.t.values, self.data.i.values, self.args)
                            for fitfun in self.fitfuns]
        self.cell = self.get_cell()

    def evaluate_fitness(self, candidate, args):
        v_candidate, t_candidate, _ = self.simulate_cell(candidate)
        vars_to_fit = [fitfun(v_candidate, t_candidate, self.simulation_params['i_inj'], self.args)
                       for fitfun in self.fitfuns]
        fitness = 0
        for i in range(len(vars_to_fit)):
            if vars_to_fit[i] is None:
                fitness = 10000
                break
            else:
                fitness += self.fitnessweights[i] * self.errfun(vars_to_fit[i], self.data_to_fit[i])
        return fitness

    def get_cell(self):
        cell = Cell.from_modeldir(self.model_dir)
        cell.insert_mechanisms(self.variable_keys)
        return cell

    def update_cell(self, candidate):
        for i in range(len(candidate)):
            for path in self.variable_keys[i]:
                self.cell.update_attr(path, candidate[i])

    def simulate_cell(self, candidate):
        self.update_cell(candidate)
        v_candidate, t_candidate, i_inj = iclamp_handling_onset(self.cell, **self.simulation_params)
        return v_candidate, t_candidate, i_inj

    def to_dict(self):
        return {'name': self.name, 'variable_keys': self.variable_keys, 'errfun_name': self.errfun_names,
                'fitfun_names': self.fitfun_names,
                'fitnessweights': self.fitnessweights, 'model_dir': self.model_dir, 'mechanism_dir': self.mechanism_dir,
                'data_dir': self.data_dir, 'simulation_params': self.init_simulation_params, 'args': self.args}


class HodgkinHuxleyFitterAdaptive(HodgkinHuxleyFitter):

    def __init__(self, name, variable_keys, errfun_name, fitfun_names, fitnessweights,
                 model_dir, mechanism_dir, data_dir, simulation_params=None, args=None):
        super(HodgkinHuxleyFitterAdaptive, self).__init__(name, variable_keys, errfun_name, fitfun_names,
                                                          fitnessweights, model_dir, mechanism_dir, data_dir,
                                                          simulation_params, args)

    def simulate_cell(self, candidate):
        self.update_cell(candidate)
        v_candidate, t_candidate, i_inj = iclamp_adaptive_handling_onset(self.cell, **self.simulation_params)
        return v_candidate, t_candidate, i_inj


class HodgkinHuxleyFitterSeveralData(HodgkinHuxleyFitter):

    def __init__(self, name, variable_keys, errfun_name, fitfun_names, fitnessweights,
                 model_dir, mechanism_dir, data_dirs, simulation_params=None, args=None):
        super(HodgkinHuxleyFitterSeveralData, self).__init__(name, variable_keys, errfun_name, fitfun_names,
                                                             fitnessweights, model_dir, mechanism_dir, data_dirs[0],
                                                             {}, args)
        self.data_dirs = data_dirs
        self.datas = list()
        for dir in data_dirs:
            self.datas.append(pd.read_csv(dir))
        self.init_simulation_params = simulation_params
        if simulation_params is None:
            simulation_params = {}
        if type(simulation_params) is dict:
            simulation_params = [simulation_params] * len(self.datas)

        self.simulation_params = list()
        self.datas_to_fit = list()
        for i, data in enumerate(self.datas):
            extracted_params = extract_simulation_params(data)
            self.simulation_params.append(merge_dicts(extracted_params, simulation_params[i]))
            self.datas_to_fit.append([fitfun(data.v.values, data.t.values, data.i.values, self.args)
                                    for fitfun in self.fitfuns])

    def evaluate_fitness(self, candidate, args):
        fitness = 0
        for s, simulation_params in enumerate(self.simulation_params):
            v_candidate, t_candidate, _ = self.simulate_cell(candidate, simulation_params)
            vars_to_fit = [fitfun(v_candidate, t_candidate, simulation_params['i_inj'], self.args)
                           for fitfun in self.fitfuns]
            for i in range(len(vars_to_fit)):
                if vars_to_fit[i] is None:
                    fitness = 10000
                    break
                else:
                    fitness += self.fitnessweights[i] * self.errfun(vars_to_fit[i], self.datas_to_fit[s][i])
            fitness += fitness
        return fitness

    def simulate_cell(self, candidate, simulation_params):
        self.update_cell(candidate)
        v_candidate, t_candidate, i_inj = iclamp_handling_onset(self.cell, **simulation_params)
        return v_candidate, t_candidate, i_inj

    def to_dict(self):
        return {'name': self.name, 'variable_keys': self.variable_keys, 'errfun_name': self.errfun_names,
                'fitfun_names': self.fitfun_names,
                'fitnessweights': self.fitnessweights, 'model_dir': self.model_dir, 'mechanism_dir': self.mechanism_dir,
                'data_dirs': self.data_dirs, 'simulation_params': self.init_simulation_params, 'args': self.args}


class HodgkinHuxleyFitterSeveralDataSeveralFitfuns(HodgkinHuxleyFitterSeveralData):
    def __init__(self, name, variable_keys, errfun_name, fitfun_names, fitnessweights,
                 model_dir, mechanism_dir, data_dirs, simulation_params=None, args=None):
        super(HodgkinHuxleyFitterSeveralDataSeveralFitfuns, self).__init__(name, variable_keys, errfun_name,
                                                                           fitfun_names[0], fitnessweights[0],
                                                                           model_dir, mechanism_dir, [data_dirs[0]],
                                                                           {}, args)
        self.data_dirs = data_dirs
        self.datas = list()
        for dir in data_dirs:
            self.datas.append(pd.read_csv(dir))

        self.fitfun_names = fitfun_names
        self.fitfuns = [[getattr(fitfuns, fitfun_name)
                         for fitfun_name in fitfun_name_set]
                        for fitfun_name_set in fitfun_names]
        self.fitnessweights = fitnessweights

        self.init_simulation_params = simulation_params
        if simulation_params is None:
            simulation_params = {}
        if type(simulation_params) is dict:
            simulation_params = [simulation_params] * len(self.datas)

        self.simulation_params = list()
        self.datas_to_fit = list()
        for i, data in enumerate(self.datas):
            extracted_params = extract_simulation_params(data)
            self.simulation_params.append(merge_dicts(extracted_params, simulation_params[i]))
            self.datas_to_fit.append([fitfun(data.v.values, data.t.values, data.i.values, self.args)
                                      for fitfun in self.fitfuns[i]])

    def evaluate_fitness(self, candidate, args):
        fitness = 0

        for s, simulation_params in enumerate(self.simulation_params):
            v_candidate, t_candidate, _ = self.simulate_cell(candidate, simulation_params)
            vars_to_fit = [fitfun(v_candidate, t_candidate, simulation_params['i_inj'], self.args)
                           for fitfun in self.fitfuns[s]]

            for i in range(len(vars_to_fit)):
                if vars_to_fit[i] is None:
                    fitness = 100000
                    break
                else:
                    fitness += self.fitnessweights[s][i] * self.errfun(vars_to_fit[i], self.datas_to_fit[s][i])
        return fitness


class HodgkinHuxleyFitterSeveralDataAdaptive(HodgkinHuxleyFitterSeveralData):

    def __init__(self, name, variable_keys, errfun_name, fitfun_names, fitnessweights,
                 model_dir, mechanism_dir, data_dirs, simulation_params=None, args=None):
        super(HodgkinHuxleyFitterSeveralDataAdaptive, self).__init__(name, variable_keys, errfun_name, fitfun_names,
                                                             fitnessweights, model_dir, mechanism_dir, data_dirs,
                                                             simulation_params, args)

    def simulate_cell(self, candidate, simulation_params):
        self.update_cell(candidate)
        v_candidate, t_candidate, i_inj = iclamp_adaptive_handling_onset(self.cell, **simulation_params)
        return v_candidate, t_candidate, i_inj


class HodgkinHuxleyFitterPareto(HodgkinHuxleyFitter):

    def __init__(self, name, variable_keys, errfun_name, fitfun_names, fitnessweights,
                 model_dir, mechanism_dir, data_dir, simulation_params=None, args=None):
        super(HodgkinHuxleyFitterPareto, self).__init__(name, variable_keys, errfun_name, fitfun_names,
                                                                fitnessweights, model_dir, mechanism_dir, data_dir,
                                                                simulation_params, args)

    def evaluate_fitness(self, candidate, args):
        fitnesses = list()
        for i, fitfun in enumerate(self.fitfuns):
            fitnesses.append(self.evaluate_fitfun(candidate, fitfun, self.fitnessweights[i], self.data_to_fit[i]))
        return Pareto(fitnesses)

    def evaluate_fitfun(self, candidate, fitfun, fitnessweight, data_to_fit):
        v_candidate, t_candidate, _ = self.simulate_cell(candidate)
        var_to_fit = fitfun(v_candidate, t_candidate, self.simulation_params['i_inj'], self.args)
        if var_to_fit is None:
            return 1000 #float("inf") TODO
        fitness = fitnessweight * self.errfun(var_to_fit, data_to_fit)
        return fitness


class HodgkinHuxleyFitterWithFitfunList(HodgkinHuxleyFitter):

    def __init__(self, name, variable_keys, errfun_name, fitfun_names, fitnessweights,
                 model_dir, mechanism_dir, data_dir, simulation_params=None, args=None):
        super(HodgkinHuxleyFitterWithFitfunList, self).__init__(name, variable_keys, errfun_name, fitfun_names,
                                                                fitnessweights, model_dir, mechanism_dir, data_dir,
                                                                simulation_params, args)

    def evaluate_fitness(self):
        evaluate_fitfuns = list()
        for i, fitfun in enumerate(self.fitfuns):
            evaluate_fitfuns.append(functools.partial(self.evaluate_fitfun, fitfun=fitfun,
                                                      fitnessweight=self.fitnessweights[i],
                                                      data_to_fit=self.data_to_fit[i]))
        return evaluate_fitfuns

    def evaluate_fitfun(self, candidate, fitfun, fitnessweight, data_to_fit):
        v_candidate, t_candidate, _ = self.simulate_cell(candidate)
        var_to_fit = fitfun(v_candidate, t_candidate, self.simulation_params['i_inj'], self.args)
        if var_to_fit is None:
            return 1000 #float("inf") TODO
        fitness = fitnessweight * self.errfun(var_to_fit, data_to_fit)
        return fitness


class HodgkinHuxleyFitterCurrentPenalty(HodgkinHuxleyFitter):

    def __init__(self, name, variable_keys, errfun_name, fitfun_names, fitnessweights,
                     model_dir, mechanism_dir, data_dir, simulation_params=None, args=None):
        super(HodgkinHuxleyFitterCurrentPenalty, self).__init__(name, variable_keys, errfun_name, fitfun_names,
                                                                fitnessweights, model_dir, mechanism_dir, data_dir,
                                                                simulation_params, args)

    def evaluate_fitness(self, candidate, args):

        v_candidate, t_candidate, currents = self.simulate_cell_with_currents(candidate)
        vars_to_fit = [fitfun(v_candidate, t_candidate, self.simulation_params['i_inj'], self.args)
                       for fitfun in self.fitfuns]
        fitness = 0
        for i in range(len(vars_to_fit)):
            if vars_to_fit[i] is None:
                fitness = 10000
                break
            else:
                fitness += self.fitnessweights[i] * self.errfun(vars_to_fit[i], self.data_to_fit[i])
        current_penalty = np.sum(np.sum(np.abs(currents))) / np.size(currents)
        fitness += current_penalty
        return fitness

    def simulate_cell_with_currents(self, candidate):
        channel_list = get_channel_list(self.cell, 'soma')
        ion_list = get_ionlist(channel_list)

        # record currents
        currents = np.zeros(len(channel_list), dtype=object)
        for i in range(len(channel_list)):
            currents[i] = self.cell.soma.record_from(channel_list[i], 'i' + ion_list[i], pos=.5)

        v_candidate, t_candidate, _ = self.simulate_cell(candidate)
        currents = [np.array(c) for c in currents]
        return v_candidate, t_candidate, currents
import pandas as pd
import numpy as np
import copy
from nrn_wrapper import Cell, load_mechanism_dir, iclamp
from optimization import errfuns
from optimization import fitfuns
from optimization.simulate import extract_simulation_params
import functools
from inspyred.ec.emo import Pareto

__author__ = 'caro'


def iclamp_handling_onset(cell, **simulation_params):
    if 'onset' in simulation_params:
        onset = simulation_params['onset']
        simulation_params_tmp = copy.copy(simulation_params)
        del simulation_params_tmp['onset']
        simulation_params_tmp['tstop'] += onset
        len_onset_idx = int(round(onset / simulation_params_tmp['dt']))
        simulation_params_tmp['i_inj'] = np.concatenate((np.zeros(len_onset_idx), simulation_params_tmp['i_inj']))

        v_candidate, t_candidate = iclamp(cell, **simulation_params_tmp)

        real_start = int(round(onset / simulation_params['dt']))
        return v_candidate[real_start:], t_candidate[:-real_start], simulation_params['i_inj']
    else:
        v_candidate, t_candidate = iclamp(cell, **simulation_params)
        return v_candidate, t_candidate, simulation_params['i_inj']


class HodgkinHuxleyFitter(object):

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
        v_candidate, t_candidate, _ = self.simulate_cell(candidate)
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
            return 1000  #float("inf")
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
        return {'variable_keys': self.variable_keys, 'errfun_name': self.errfun_names, 'fitfun_names': self.fitfun_names,
                'fitnessweights': self.fitnessweights, 'model_dir': self.model_dir, 'mechanism_dir': self.mechanism_dir,
                'data_dir': self.data_dir, 'simulation_params': self.init_simulation_params, 'args': self.args}


class HodgkinHuxleyFitterSeveralData(HodgkinHuxleyFitter):

    def __init__(self, variable_keys, errfun_name, fitfun_names, fitnessweights,
                 model_dir, mechanism_dir, data_dirs, simulation_params=None, args=None):
        super(HodgkinHuxleyFitterSeveralData, self).__init__(variable_keys, errfun_name, fitfun_names,
                                                                fitnessweights, model_dir, mechanism_dir, data_dirs[0],
                                                                simulation_params, args)
        self.data_dirs = data_dirs
        self.datas = list()
        for dir in data_dirs:
            self.datas.append(pd.read_csv(dir))
        self.init_simulation_params = simulation_params
        if simulation_params is None:
            simulation_params = {}
        self.simulation_params = list()
        self.datas_to_fit = list()
        for data in self.datas:
            self.simulation_params.append(extract_simulation_params(data, **simulation_params))
            self.datas_to_fit.append([fitfun(data.v.values, data.t.values, data.i.values, self.args)
                                    for fitfun in self.fitfuns])

    def evaluate_fitness(self, candidate, args):
        fitness_total = 0
        for s, simulation_params in enumerate(self.simulation_params):
            v_candidate, t_candidate, _ = self.simulate_cell(candidate, simulation_params)
            vars_to_fit = [fitfun(v_candidate, t_candidate, simulation_params['i_inj'], self.args)
                           for fitfun in self.fitfuns]
            num_nones = 0
            fitness = 0
            for i in range(len(vars_to_fit)):
                if vars_to_fit[i] is None:
                    num_nones += 1
                else:
                    fitness += self.fitnessweights[i] * self.errfun(vars_to_fit[i], self.datas_to_fit[s][i])
            if num_nones == len(vars_to_fit):
                fitness = 1000  # float("inf")
            fitness_total += fitness
        return fitness_total

    def simulate_cell(self, candidate, simulation_params):
        self.update_cell(candidate)
        v_candidate, t_candidate, i_inj = iclamp_handling_onset(self.cell, **simulation_params)
        return v_candidate, t_candidate, i_inj

    def to_dict(self):
        return {'variable_keys': self.variable_keys, 'errfun_name': self.errfun_names, 'fitfun_names': self.fitfun_names,
                'fitnessweights': self.fitnessweights, 'model_dir': self.model_dir, 'mechanism_dir': self.mechanism_dir,
                'data_dirs': self.data_dirs, 'simulation_params': self.init_simulation_params, 'args': self.args}


class HodgkinHuxleyFitterPareto(HodgkinHuxleyFitter):

    def __init__(self, variable_keys, errfun_name, fitfun_names, fitnessweights,
                 model_dir, mechanism_dir, data_dir, simulation_params=None, args=None):
        super(HodgkinHuxleyFitterPareto, self).__init__(variable_keys, errfun_name, fitfun_names,
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

    def __init__(self, variable_keys, errfun_name, fitfun_names, fitnessweights,
                 model_dir, mechanism_dir, data_dir, simulation_params=None, args=None):
        super(HodgkinHuxleyFitterWithFitfunList, self).__init__(variable_keys, errfun_name, fitfun_names,
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
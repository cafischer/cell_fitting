import numpy as np
import pandas as pd
from nrn_wrapper import load_mechanism_dir, Cell, iclamp

from cell_fitting.optimization.helpers import get_channel_list, get_ionlist, get_cellarea
from cell_fitting.util import convert_from_unit
from cell_fitting.optimization.linear_regression import linear_regression
from cell_fitting.optimization.simulate import currents_given_v
from cell_fitting.optimization.simulate import extract_simulation_params

__author__ = 'caro'


class LinearRegressionFitter(object):

    def __init__(self, name, variable_keys, model_dir, mechanism_dir, data_dir, simulation_params, with_cm=True):

        self.name = name
        self.variable_keys = variable_keys  # variables to fit (not gbars!)
        self.model_dir = model_dir
        self.mechanism_dir = mechanism_dir
        if mechanism_dir is not None:
            load_mechanism_dir(mechanism_dir)
        self.data_dir = data_dir
        self.data = pd.read_csv(data_dir)
        self.init_simulation_params = simulation_params
        self.simulation_params = extract_simulation_params(self.data, **simulation_params)
        self.with_cm = with_cm

        # create cell and set gbars to one
        self.cell = self.get_cell()
        self.channel_list = get_channel_list(self.cell, 'soma')
        self.ion_list = get_ionlist(self.channel_list)

        # compute dvdt
        self.v_exp = self.data.v.values
        self.t_exp = self.data.t.values
        dt = self.t_exp[1] - self.t_exp[0]
        self.dvdt = np.concatenate((np.array([(self.v_exp[1] - self.v_exp[0]) / dt]), np.diff(self.v_exp) / dt))  #V/s

        # compute cell areas
        self.cell_area = get_cellarea(convert_from_unit('u', self.cell.soma.L),
                                      convert_from_unit('u', self.cell.soma.diam))  # m**2

        # compute injected current
        self.i_inj = convert_from_unit('n', self.data.i.values)  # A

    def evaluate_fitness(self, candidate, args):
        self.update_cell(candidate)
        self.update_gbar(np.ones(len(self.channel_list)))
        currents = self.get_currents()
        weights, residual = self.do_linear_regression(currents)
        return residual * 1e7  # TODO
        #self.update_gbar(weights)
        #v, t, i = self.simulate_cell()
        #return rms(v, self.data.v.values)

    def get_v(self, candidate):
        self.update_cell(candidate)
        self.update_gbar(np.ones(len(self.channel_list)))
        currents = self.get_currents()
        weights, residual = self.do_linear_regression(currents)
        self.update_gbar(weights)
        v, t, i = self.simulate_cell()
        return v

    def update_cell(self, candidate):
        for i in range(len(candidate)):
            for path in self.variable_keys[i]:
                self.cell.update_attr(path, candidate[i])

    def get_currents(self):
        currents = currents_given_v(self.v_exp, self.t_exp, self.cell.soma, self.channel_list, self.ion_list,
                                    self.simulation_params['celsius'])
        currents = convert_from_unit('da', currents) * self.cell_area  # A
        return currents

    def do_linear_regression(self, currents):
        if self.with_cm:
            weights, _, residual, y, X = linear_regression(self.dvdt, self.i_inj, currents, i_pas=0,
                                                           Cm=None, cell_area=self.cell_area)
        else:
            Cm = convert_from_unit('c', self.cell.soma.cm) * self.cell_area  # F
            weights, residual, y, X = linear_regression(self.dvdt, self.i_inj, currents, i_pas=0, Cm=Cm)

        return weights, residual

    def update_gbar(self, weights):
        for i, channel in enumerate(self.channel_list):
            if channel == 'pas':
                self.cell.update_attr(['soma', '0.5', channel, 'g'], weights[i])
            else:
                self.cell.update_attr(['soma', '0.5', channel, 'gbar'], weights[i])

    def get_cell(self):
        cell = Cell.from_modeldir(self.model_dir)
        cell.insert_mechanisms(self.variable_keys)
        return cell

    def simulate_cell(self):
        v_candidate, t_candidate = iclamp(self.cell, **self.simulation_params)
        return v_candidate, t_candidate, self.simulation_params['i_inj']

    def to_dict(self):
        return {'name': self.name, 'variable_keys': self.variable_keys, 'model_dir': self.model_dir, 'mechanism_dir': self.mechanism_dir,
                'data_dir': self.data_dir, 'simulation_params': self.init_simulation_params, 'with_cm': self.with_cm}
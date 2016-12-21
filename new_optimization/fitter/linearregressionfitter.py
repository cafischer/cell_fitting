import pandas as pd
import numpy as np
from optimization.linear_regression import *
from nrn_wrapper import *
from optimization.simulate import currents_given_v
from optimization import errfuns
from optimization import fitfuns
from optimization.simulate import extract_simulation_params
import functools

__author__ = 'caro'


class LinearRegressionFitter(object):

    def __init__(self, variable_keys, model_dir, mechanism_dir, data_dir, simulation_params, with_cm=True):

        self.variable_keys = variable_keys  # variables to fit (not gbars!)
        self.model_dir = model_dir
        self.mechanism_dir = mechanism_dir
        if mechanism_dir is not None:
            load_mechanism_dir(mechanism_dir)
        self.data_dir = data_dir
        self.data = pd.read_csv(data_dir)
        self.simulation_params = simulation_params
        self.with_cm = with_cm

        # create cell and set gbars to one
        self.cell = self.get_cell()
        self.channel_list = get_channel_list(self.cell, 'soma')
        self.ion_list = get_ionlist(self.channel_list)
        for channel in self.channel_list:
            if channel == 'pas':
                self.cell.update_attr(['soma', '0.5', channel, 'g'], 1)
            else:
                self.cell.update_attr(['soma', '0.5', channel, 'gbar'], 1)

        # compute dvdt
        self.v_exp = self.data.v.values
        self.t_exp = self.data.t.values
        dt = self.t_exp[1] - self.t_exp[0]
        self.dvdt = np.concatenate((np.array([(self.v_exp[1] - self.v_exp[0]) / dt]), np.diff(self.v_exp) / dt))  #V/s

        # compute cell areas
        self.cell_area = get_cellarea(convert_unit_prefix('u', self.cell.soma.L),
                                 convert_unit_prefix('u', self.cell.soma.diam))  # m**2

        # compute injected current
        self.i_inj = convert_unit_prefix('n', self.data.i.values)  # A

    def evaluate_fitness(self, candidate, args):

        # update cell
        self.update_cell(candidate)

        # get current traces
        currents = currents_given_v(self.v_exp, self.t_exp, self.cell.soma, self.channel_list, self.ion_list,
                                    self.simulation_params['celsius'])
        currents = convert_unit_prefix('da', currents) * self.cell_area  # A

        # linear regression
        if self.with_cm:
            weights_adjusted, weights, residual, y, X = linear_regression(self.dvdt, self.i_inj, currents, i_pas=0,
                                                                          Cm=None, cell_area=self.cell_area)
        else:
            Cm = convert_unit_prefix('c', self.cell.soma.cm) * self.cell_area  # F
            weights, residual, y, X = linear_regression(self.dvdt, self.i_inj, currents, i_pas=0, Cm=Cm)

        return residual

    def get_cell(self):
        cell = Cell.from_modeldir(self.model_dir)
        cell.insert_mechanisms(self.variable_keys)
        return cell

    def update_cell(self, candidate):
        for i in range(len(candidate)):
            for path in self.variable_keys[i]:
                self.cell.update_attr(path, candidate[i])

    def to_dict(self):
        return {'variable_keys': self.variable_keys, 'model_dir': self.model_dir, 'mechanism_dir': self.mechanism_dir,
                'data_dir': self.data_dir, 'simulation_params': self.simulation_params, 'with_cm': self.with_cm}
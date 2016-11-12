import numpy as np

from new_optimization.fitter.hodgkinhuxleyfitter import HodgkinHuxleyFitter
from optimization.helpers import get_channel_list, get_ionlist, get_cellarea, convert_unit_prefix
from optimization.simulate import currents_given_v
from nrn_wrapper import iclamp

__author__ = 'caro'


class Model:

    def __init__(self, fitter_params):

        self.fitter = HodgkinHuxleyFitter(**fitter_params)
        self.lhsHH = self.get_lhsHH()
        self.channel_list = get_channel_list(self.fitter.cell, 'soma')
        self.ion_list = get_ionlist(self.channel_list)

    def get_lhsHH(self):
        dt = self.fitter.simulation_params['dt']
        v_exp = self.fitter.data.v.values
        dvdt = np.concatenate((np.array([(v_exp[1]-v_exp[0])/dt]), np.diff(v_exp) / dt))  # TODO: check again if this is right

        # convert units
        cell_area = get_cellarea(convert_unit_prefix('u', self.fitter.cell.soma.L),
                                 convert_unit_prefix('u', self.fitter.cell.soma.diam))  # m
        Cm = convert_unit_prefix('c', self.fitter.cell.soma.cm) * cell_area  # F
        i_inj = convert_unit_prefix('n', self.fitter.data.i.values)  # A
        dvdt = convert_unit_prefix('m', dvdt)  # V

        return dvdt * Cm - i_inj  # A

    def get_rhsHH(self, currents):
        return -1 * np.sum(currents, 0)

    def get_current(self):
        # generate current traces
        currents = currents_given_v(self.fitter.data.v.values, self.fitter.data.t.values, self.fitter.cell.soma,
                                    self.channel_list, self.ion_list, self.fitter.simulation_params['celsius'])

        # convert units
        cell_area = get_cellarea(convert_unit_prefix('u', self.fitter.cell.soma.L),
                                 convert_unit_prefix('u', self.fitter.cell.soma.diam))  # m
        Cm = convert_unit_prefix('c', self.fitter.cell.soma.cm) * cell_area  # F
        currents = convert_unit_prefix('da', currents) * cell_area  # A

        return currents

    def simulate(self):
        return iclamp(self.fitter.cell, **self.fitter.simulation_params)

    def update_var(self, id, value):
        for path in self.fitter.variable_keys[id]:
            self.fitter.cell.update_attr(path, value)
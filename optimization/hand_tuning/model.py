import numpy as np

from optimization import problems
from optimization.problems import get_channel_list, get_ionlist, get_cellarea, convert_unit_prefix
from optimization.simulate import currents_given_v, run_simulation

__author__ = 'caro'


class Model:

    def __init__(self, problem_dict):

        self.problem = getattr(problems, problem_dict['name'])(**problem_dict)
        self.lhsHH = self.get_lhsHH()
        self.channel_list = get_channel_list(self.problem.cell, 'soma')
        self.ion_list = get_ionlist(self.channel_list)

    def get_lhsHH(self):
        dt = self.problem.simulation_params['dt']
        v_exp = self.problem.data.v.values
        dvdt = np.concatenate((np.array([(v_exp[1]-v_exp[0])/dt]), np.diff(v_exp) / dt))  # TODO: check again if this is right

        # convert units
        cell_area = get_cellarea(convert_unit_prefix('u', self.problem.cell.soma.L),
                                 convert_unit_prefix('u', self.problem.cell.soma.diam))  # m
        Cm = convert_unit_prefix('c', self.problem.cell.soma.cm) * cell_area  # F
        i_inj = convert_unit_prefix('n', self.problem.data.i.values)  # A
        dvdt = convert_unit_prefix('m', dvdt)  # V

        return dvdt * Cm - i_inj  # A

    def get_rhsHH(self, currents):
        return -1 * np.sum(currents, 0)

    def get_current(self):
        # generate current traces
        currents = currents_given_v(self.problem.data.v.values, self.problem.data.t.values, self.problem.cell.soma,
                                    self.channel_list, self.ion_list, self.problem.simulation_params['celsius'])

        # convert units
        cell_area = get_cellarea(convert_unit_prefix('u', self.problem.cell.soma.L),
                                 convert_unit_prefix('u', self.problem.cell.soma.diam))  # m
        Cm = convert_unit_prefix('c', self.problem.cell.soma.cm) * cell_area  # F
        currents = convert_unit_prefix('da', currents) * cell_area  # A

        return currents

    def simulate(self):
        return run_simulation(self.problem.cell, **self.problem.simulation_params)

    def update_var(self, id, value):
        for path in self.problem.path_variables[id]:
            self.problem.cell.update_attr(path, value)
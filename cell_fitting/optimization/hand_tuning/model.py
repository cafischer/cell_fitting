#from new_optimization.fitter.hodgkinhuxleyfitter import HodgkinHuxleyFitter
from cell_fitting.new_optimization.fitter import FitterFactory
from cell_fitting.optimization.helpers import *
from cell_fitting.optimization.simulate import currents_given_v, iclamp_handling_onset, iclamp_adaptive_handling_onset

__author__ = 'caro'


class Model:

    def __init__(self, fitter_params):

        self.fitter = FitterFactory().make_fitter(fitter_params)  #HodgkinHuxleyFitter(**fitter_params)
        self.lhsHH = self.get_lhsHH()
        self.channel_list = get_channel_list(self.fitter.cell, 'soma')
        self.ion_list = get_ionlist(self.channel_list)

    def get_lhsHH(self):
        dt = self.fitter.simulation_params['dt']  # ms
        v_exp = self.fitter.data.v.values  # mV
        dvdt = np.concatenate((np.array([(v_exp[1]-v_exp[0])/dt]), np.diff(v_exp) / dt))  # V/m

        # convert units
        cell_area = get_cellarea(convert_from_unit('u', self.fitter.cell.soma.L),
                                 convert_from_unit('u', self.fitter.cell.soma.diam))  # m**2
        Cm = convert_from_unit('c', self.fitter.cell.soma.cm) * cell_area  # F
        i_inj = convert_from_unit('n', self.fitter.data.i.values)  # A

        return dvdt * Cm - i_inj  # A

    def get_rhsHH(self, currents):
        return -1 * np.sum(currents, 0)

    def get_current(self):
        # generate current traces
        currents = currents_given_v(self.fitter.data.v.values, self.fitter.data.t.values, self.fitter.cell.soma,
                                    self.channel_list, self.ion_list, self.fitter.simulation_params['celsius'])  # mA/cm**2

        # convert units
        cell_area = get_cellarea(convert_from_unit('u', self.fitter.cell.soma.L),
                                 convert_from_unit('u', self.fitter.cell.soma.diam))  # m**2
        currents = convert_from_unit('da', currents) * cell_area  # A

        return currents

    def simulate(self):
        if 'continuous' in self.fitter.simulation_params:  # TODO
            v, t, i_inj = iclamp_adaptive_handling_onset(self.fitter.cell, **self.fitter.simulation_params)
        else:
            v, t, i_inj = iclamp_handling_onset(self.fitter.cell, **self.fitter.simulation_params)
        return v, t

    def update_var(self, id, value):
        for path in self.fitter.variable_keys[id]:
            self.fitter.cell.update_attr(path, value)
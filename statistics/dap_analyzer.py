from __future__ import division
import numpy as np
import pylab as pl
from scipy.signal import argrelmin, argrelmax
import warnings

__author__ = 'caro'


class ApAnalyzer:

    def __init__(self, v, t):
        self.v = v
        self.t = t

    def get_AP_onsets(self, threshold=-45, vrest=-65):
        """
        Returns the indices of the times where the membrane potential crossed threshold.
        :param threshold: AP threshold.
        :type threshold: float
        :param vrest: Resting potential.
        :type vrest: float
        :return: Indices of the times where the membrane potential crossed threshold.
        :rtype: int
        """
        if threshold < vrest:
            warnings.warn('Action potential threshhold is lower than resting potential', UserWarning)

        return np.nonzero(np.diff(np.sign(self.v-threshold)) == 2)[0]

    def get_AP_max(self, AP_onset, AP_end, order=5, interval=None):
        """
        Returns the index of the local maximum of the AP between AP onset and end during dur.
        :param AP_onset: Index where the membrane potential crosses the AP threshold
        :type AP_onset: int
        :param AP_end: Index of the end of the AP (e.g. delimited by the onset of the next AP or the end of the trace)
        :type AP_end: int
        :param order: Number of points to consider for determining the local maxima.
        :type order: int
        :param interval: Length of the interval during which the maximum of the AP shall occur starting from AP onset.
        :type interval: int
        :return: Index of the Maximum of the AP.
        :rtype: int
        """

        maxima = argrelmax(self.v[AP_onset:AP_end], order=order)[0]
        if interval is not None:
            maxima = maxima[maxima < interval]

        if np.size(maxima) == 0:
            return None
        else:
            return maxima[np.argmax(self.v[AP_onset:AP_end][maxima])] + AP_onset

    def get_fAHP_min(self, AP_max, AP_end, order=5, interval=None):
        """
        Returns the index of the local minimum found after AP maximum.
        :param AP_max: Index of the maximum of the AP.
        :type AP_max: int
        :param AP_end: Index of the end of the AP (e.g. delimited by the onset of the next AP or the end of the trace)
        :type AP_end: int
        :param order: Number of points to consider for determining the local minima.
        :type order: int
        :param interval: Length of the interval during which the minimum of the fAHP shall occur starting from AP max.
        :type interval: int
        :return: Index of the Minimum of the fAHP.
        :rtype: int
        """

        minima = argrelmin(self.v[AP_max:AP_end], order=order)[0]
        if interval is not None:
            minima = minima[minima < interval]

        if np.size(minima) == 0:
            return None
        else:
            return minima[np.argmin(self.v[AP_max:AP_end][minima])] + AP_max

    def get_DAP_max(self, fAHP_min, AP_end, order=5, interval=None):
        """
        Returns the index of the local maximum found after fAHP.
        :param fAHP_min: Index of the minimum of the fAHP.
        :type fAHP_min: int
        :param AP_end: Index of the end of the AP (e.g. delimited by the onset of the next AP or the end of the trace)
        :type AP_end: int
        :param order: Number of points to consider for determining the local minima.
        :type order: int
        :param interval: Length of the interval during which the minimum of the fAHP shall occur starting from AP max.
        :type interval: int
        :return: Index of maximum of the DAP.
        :rtype: int
        """
        maxima = argrelmax(self.v[fAHP_min:AP_end], order=order)[0]
        if interval is not None:
            maxima = maxima[maxima < interval]

        if np.size(maxima) == 0:
            return None
        else:
            return maxima[np.argmax(self.v[fAHP_min:AP_end][maxima])] + fAHP_min

    def get_AP_amp(self, AP_max, vrest):
        return self.v[AP_max] - vrest

    def get_AP_width(self, AP_onset, AP_max, AP_end):
        halfmax =(self.v[AP_max] - vrest)/2
        width1 = np.argmin(np.abs(self.v[AP_onset:AP_max]-vrest-halfmax)) + AP_onset
        width2 = np.argmin(np.abs(self.v[AP_max:AP_end]-vrest-halfmax)) + AP_max
        return self.t[width2] - self.t[width1]

    def get_DAP_amp(self, DAP_max, vrest):
        return self.v[DAP_max] - vrest

    def get_DAP_deflection(self, DAP_max, fAHP_min):
        return self.v[DAP_max] - self.v[fAHP_min]

    def get_DAP_width(self, fAHP_min, DAP_max, AP_end, vrest):
        halfmax = (self.v[fAHP_min] - vrest)/2
        halfwidth = np.nonzero(np.diff(np.sign(self.v[DAP_max:AP_end]-vrest-halfmax)) == -2)[0][0] + DAP_max
        return self.t[halfwidth] - self.t[fAHP_min]


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from fit_currents.error_analysis.model_generator import *
    from model.cell_builder import *
    from optimization.fitfuns import run_simulation
    from optimization.optimizer import extract_simulation_params
    from fit_currents.vclamp import *
    from utilities import merge_dicts

    # # test on experimental data
    data_dir = '../data/new_cells/2015_08_11d/dap/dap.csv'
    data = pd.read_csv(data_dir)
    v_exp = np.array(data.v)
    i_exp = np.array(data.i)
    t_exp = np.array(data.t)
    dt_exp = t_exp[1]-t_exp[0]
    vrest = np.mean(v_exp[:100])

    dap_analyzer = ApAnalyzer(v_exp, t_exp)
    AP_onsets = dap_analyzer.get_AP_onsets()
    AP_onset = AP_onsets[0]
    AP_end = -1

    AP_max = dap_analyzer.get_AP_max(AP_onset, AP_end, interval=1/dt_exp)
    fAHP_min = dap_analyzer.get_fAHP_min(AP_max, AP_end, interval=5/dt_exp)
    DAP_max = dap_analyzer.get_DAP_max(fAHP_min, AP_end, interval=10/dt_exp)

    AP_amp = dap_analyzer.get_AP_amp(AP_max, vrest)
    AP_width, w1, w2 = dap_analyzer.get_AP_width(AP_onset, AP_max, AP_end)
    DAP_amp = dap_analyzer.get_DAP_amp(DAP_max, vrest)
    DAP_deflection = dap_analyzer.get_DAP_deflection(DAP_max, fAHP_min)
    DAP_width = dap_analyzer.get_DAP_width(fAHP_min, DAP_max, AP_end, vrest)
    print 'AP amplitude: ' + str(AP_amp)
    print 'AP width: ' + str(AP_width)
    print 'DAP amplitude: ' + str(DAP_amp)
    print 'DAP deflection: ' + str(DAP_deflection)
    print 'DAP width: ' + str(DAP_width)

    pl.figure()
    pl.plot(t_exp, v_exp, 'k', label='V')
    pl.plot(t_exp[AP_onsets], v_exp[AP_onsets], 'or', label='AP onsets')
    pl.plot(t_exp[AP_max], v_exp[AP_max], 'ob', label='AP maximum')
    pl.plot(t_exp[fAHP_min], v_exp[fAHP_min], 'oy', label='fAHP minimum')
    pl.plot(t_exp[DAP_max], v_exp[DAP_max], 'og', label='DAP maximum')
    pl.legend()
    pl.show()

    # # test on randomly generated models
    n_models = 5
    data_dir = '../data/cell_2013_12_13f/dap/dap1.csv'
    protocol = 'stepramp'
    h.nrn_load_dll(complete_mechanismdir("../model/channels_currentfitting"))
    h.nrn_load_dll(complete_mechanismdir("../model/channels_vclamp"))
    all_channels = ['narsg', 'naf', 'na8st', 'nap', 'ka', 'kdr', 'ih_slow', 'ih_fast', 'passive']
    E_ion = {'ena': 60, 'ek': -80, 'eca': 80, 'ehcn': -20, 'epas': -70}
    all_channel_boundaries = [["g_narsg", 0, 0.1, [["soma", "mechanisms", "narsg", "gbar"]]],
                                ["g_naf", 0, 0.1, [["soma", "mechanisms", "naf", "gbar"]]],
                                ["g_na8st", 0, 0.1, [["soma", "mechanisms", "na8st", "gbar"]]],
                                ["g_nap", 0, 0.1, [["soma", "mechanisms", "nap", "gbar"]]],
                                ["g_ka", 0, 0.1, [["soma", "mechanisms", "ka", "gbar"]]],
                                ["g_kdr", 0, 0.1, [["soma", "mechanisms", "kdr", "gbar"]]],
                                ["g_ih_slow", 0, 0.1, [["soma", "mechanisms", "ih_slow", "gbar"]]],
                                ["g_ih_fast", 0, 0.1, [["soma", "mechanisms", "ih_fast", "gbar"]]],
                                ["g_passive", 0, 0.1, [["soma", "mechanisms", "passive", "gbar"]]]
                                ]

    model_params = {
        "soma": {
            "geom": {
                "diam": 8.0,
                "L": 16.0
            },
            "Ra": 100,
            "cm": 1.0
        },
        "celsius": 35
    }

    data = pd.read_csv(data_dir)
    dt = data.t[1]-data.t[0]
    t = np.array(data.t)
    i_inj = np.array(data.i)

    for i in range(n_models):
        # generate models
        mask = np.random.rand(len(all_channels)) > 0.5
        channel_list = np.array(all_channels)[mask]
        ion_list = get_ionlist(channel_list)
        channel_boundaries = np.array(all_channel_boundaries, dtype=object)[mask]

        model, vals = generate_model(channel_boundaries)
        model = merge_dicts(model, model_params)
        weights_model = {k: vals[j] for j, k in enumerate(channel_list)}

        cell = Cell(model)
        cell_area = cell.soma(.5).area() * 1e-8
        cell = set_Epotential(cell, E_ion)
        simulation_params = extract_simulation_params(data)
        v_model, _ = run_simulation(cell, **simulation_params)
        v_model = np.array(v_model)

        # analyze voltage trace
        dap_analyzer = ApAnalyzer(v_model, t)
        v_rest = np.mean(v_model[:100])
        AP_onsets = dap_analyzer.get_AP_onsets(vrest=v_rest)
        if len(AP_onsets) == 0:
            print 'No APs!'
        else:
            AP_onset = AP_onsets[0]
            if len(AP_onsets) == 1:
                AP_end = -1
            elif len(AP_onsets) > 1:
                AP_end = AP_onsets[1]
            AP_max = dap_analyzer.get_AP_max(AP_onset, AP_end, interval=1/dt_exp)
            if AP_max is not None:
                fAHP_min = dap_analyzer.get_fAHP_min(AP_max, AP_end, interval=5/dt)
            else:
                fAHP_min = None
                DAP_max = None
            if fAHP_min is not None:
                DAP_max = dap_analyzer.get_DAP_max(fAHP_min, AP_end, interval=10/dt)
            else:
                DAP_max = None

            pl.figure()
            pl.plot(t, v_model, 'xk', label='V')
            pl.plot(t[AP_onsets], v_model[AP_onsets], 'or', label='AP onsets')
            if AP_max is not None: pl.plot(t[AP_max], v_model[AP_max], 'ob', label='AP maximum')
            if fAHP_min is not None: pl.plot(t[fAHP_min], v_model[fAHP_min], 'oy', label='fAHP minimum')
            if DAP_max is not None: pl.plot(t[DAP_max], v_model[DAP_max], 'og', label='DAP maximum')
            pl.legend()
            pl.show()
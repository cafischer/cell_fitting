from cell_fitting.new_optimization.fitter.fitter_interface import Fitter
from cell_fitting.optimization.errfuns import rms
import numpy as np
import pandas as pd
import cell_fitting.test_channels.channel_characteristics


class ChannelFitterSingleTraces(Fitter):

    def __init__(self, name, data_dir, variable_names, fixed_params, n_params, compute_current_name):
        super(ChannelFitterSingleTraces, self).__init__()
        self.name = name
        self.data_dir = data_dir
        self.variable_names = variable_names
        self.i_traces = pd.read_csv(data_dir, index_col=0)
        self.i_traces /= np.max(np.max(np.abs(self.i_traces)))  # normalize
        self.v_steps = [int(c) for c in self.i_traces.columns.values]
        self.fixed_params = fixed_params
        self.n_params = n_params
        self.compute_current_name = compute_current_name
        self.compute_current = getattr(test_channels.channel_characteristics, compute_current_name)

    def evaluate_fitness(self, candidate, args):
        i_traces_fit = self.simulate(candidate, args)

        return np.sum([rms(self.i_traces[str(v_step)], i_trace_fit)
                        for v_step, i_trace_fit in zip(self.v_steps, i_traces_fit)])

    def simulate(self, candidate, args):
        t_vec = self.i_traces.index.values
        i_traces_fit = []
        candidate = np.array(candidate).reshape(len(self.v_steps), self.n_params)
        for i, v_step in enumerate(self.v_steps):
            i_traces_fit.append(self.compute_current(v_step, t_vec, *candidate[i], **self.fixed_params))

        i_traces_fit = [i_trace_fit / np.max(np.abs(i_traces_fit)) for i_trace_fit in i_traces_fit]

        return i_traces_fit

    def to_dict(self):
        return {'name': self.name, 'data_dir': self.data_dir, 'variable_names': self.variable_names,
                'fixed_params': self.fixed_params, 'n_params': self.n_params,
                'compute_current_name': self.compute_current_name}


class ChannelFitterAllTraces(ChannelFitterSingleTraces):

    def __init__(self, name, data_dir, variable_names, fixed_params, n_params, compute_current_name):
        super(ChannelFitterAllTraces, self).__init__(name, data_dir, variable_names, fixed_params, n_params,
                                                     compute_current_name)

    def simulate(self, candidate, args):
        t_vec = self.i_traces.index.values
        i_traces_fit = []
        for i, v_step in enumerate(self.v_steps):
            i_traces_fit.append(self.compute_current(v_step, t_vec, *candidate, **self.fixed_params))

        i_traces_fit = [i_trace_fit / np.max(np.abs(i_traces_fit)) for i_trace_fit in i_traces_fit]

        return i_traces_fit

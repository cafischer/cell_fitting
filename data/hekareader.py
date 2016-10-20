from heka_reader import heka_reader
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as pl


def nested_dict():
    return defaultdict(nested_dict)


def set_key_value_for_copied_keys(dictionary, key, value, copy_num=1):
    if dictionary.get(key) is None:
        dictionary[key] = value
    elif dictionary.get(key + '(' + str(copy_num) + ')') is None:
        dictionary[key + '(' + str(copy_num) + ')'] = value
    else:
        copy_num += 1
        set_key_value_for_copied_keys(dictionary, key, value, copy_num)


class HekaReader:

    def __init__(self, file_dir):
        self.data_bundle = heka_reader.Bundle(file_dir)

    def get_type_to_index(self):
        type_to_index = nested_dict()
        for g, group in enumerate(self.data_bundle.pul.children):
            for se, series in enumerate(group.children):
                for sw, sweep in enumerate(series.children):
                    for t, trace in enumerate(sweep.children):
                        group_type = HekaReader.get_type(group)
                        series_type = HekaReader.get_type(series)
                        sweep_type = HekaReader.get_type(sweep)
                        trace_type = HekaReader.get_type(trace)
                        type_to_index[group_type][series_type][sweep_type][trace_type] = [g, se, sw, t]
        return type_to_index

    def get_protocol(self, group_type):
        protocol_to_series = dict()
        selected_group = self.get_group(group_type)
        for series in selected_group.children:
            set_key_value_for_copied_keys(protocol_to_series, HekaReader.get_label(series), HekaReader.get_type(series))
        return protocol_to_series

    def get_group(self, group_type):
        for group in self.data_bundle.pul.children:
            if HekaReader.get_type(group) == group_type:
                return group
        raise ValueError('Group does not exist!')

    @staticmethod
    def is_group(data):
        return 'Group' in HekaReader.get_type(data)

    @staticmethod
    def is_series(data):
        return 'Series' in HekaReader.get_type(data)

    @staticmethod
    def is_sweep(data):
        return 'Sweep' in HekaReader.get_type(data)

    @staticmethod
    def get_type(data):
        type = data.__class__.__name__
        if type.endswith('Record'):
            type = type[:-6]
        try:
            type += str(getattr(data, type + 'Count'))
        except AttributeError:
            pass
        return type

    @staticmethod
    def get_label(data):
        try:
            label = data.Label
        except AttributeError:
            label = ''
        return label

    def get_xy(self, indices):
        y = self.data_bundle.data[indices]
        trace = self.get_trace(indices)
        x = np.linspace(trace.XStart, trace.XStart + trace.XInterval * (len(y) - 1), len(y))
        return x, y

    def get_units_xy(self, indices):
        trace = self.get_trace(indices)
        return trace.XUnit, trace.YUnit

    def get_trace(self, indices):
        trace = self.data_bundle.pul
        for i in indices:
            trace = trace[i]
        return trace


def get_indices_for_protocol(hekareader, protocol):
    type_to_index = hekareader.get_type_to_index()
    group = 'Group1'
    protocol_to_series = hekareader.get_protocol(group)
    series = protocol_to_series[protocol]
    sweeps = ['Sweep' + str(i) for i in range(1, len(type_to_index[group][series]))]
    trace = 'Trace1'
    indices = [type_to_index[group][series][sweep][trace] for sweep in sweeps]
    return indices


if __name__ == '__main__':
    file_dir = './2015_08_26b/2015_08_26b.dat'
    hekareader = HekaReader(file_dir)
    #hekareader.data_bundle.pgf # TODO
    type_to_index = hekareader.get_type_to_index()

    group = 'Group1'
    protocol = 'rampIV'
    trace = 'Trace1'
    protocol_to_series = hekareader.get_protocol(group)
    series = protocol_to_series[protocol]
    sweeps = ['Sweep' + str(i) for i in range(1, len(type_to_index[group][series]))]

    indices = [type_to_index[group][series][sweep][trace] for sweep in sweeps]

    fig = pl.figure()
    ax = fig.add_subplot(111)
    for index in indices:
        x, y = hekareader.get_xy(index)
        x_unit, y_unit = hekareader.get_units_xy(index)

        ax.plot(x*1000, y*1000, 'k')
        ax.set_xlabel('Time ('+x_unit+')', fontsize=18)
        ax.set_ylabel('Membrane Potential (' + y_unit + ')', fontsize=18)
        ax.tick_params(labelsize=15)
    pl.tight_layout()
    pl.show()
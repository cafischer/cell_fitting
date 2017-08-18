from heka_reader import HekaReader
import pandas as pd
import os
import matplotlib.pyplot as pl
import numpy as np
from cell_fitting.data import shift_v_rest, set_v_rest
from cell_fitting.optimization.helpers import convert_to_unit


def get_v_and_t_from_heka(file_dir, protocol, group='Group1', trace='Trace1', sweep_idxs=None, return_sweep_idxs=False):
    hekareader = HekaReader(file_dir)
    type_to_index = hekareader.get_type_to_index()
    protocol_to_series = hekareader.get_protocol(group)
    series = protocol_to_series[protocol]
    sweeps = ['Sweep' + str(i) for i in range(1, len(type_to_index[group][series]) + 1)]
    # print '# sweeps: ', len(sweeps)
    if sweep_idxs is None:
        sweep_idxs = range(len(sweeps))
    sweeps = [sweeps[index] for index in sweep_idxs]
    indices = [type_to_index[group][series][sweep][trace] for sweep in sweeps]

    v = [0] * len(indices)
    t = [0] * len(indices)
    for i, index in enumerate(indices):
        t[i], v[i] = hekareader.get_xy(index)
        x_unit, y_unit = hekareader.get_units_xy(index)
        assert x_unit == 's'
        assert y_unit == 'V'
        t[i] = convert_to_unit('m', t[i])  # ms
        v[i] = convert_to_unit('m', v[i])  # mV
        t[i] = list(t[i])
        v[i] = list(v[i])

    if return_sweep_idxs:
        return np.array(v), np.array(t), sweep_idxs
    return np.array(v), np.array(t)


def get_i_inj(protocol, sweep_idxs):
    protocol_dir = '/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/data/Protocols'
    try:
        i_inj_base = pd.read_csv(os.path.join(protocol_dir, protocol + '.csv'), header=None).values[:, 0]
    except IOError:
        raise ValueError('i_inj does not exist for the given protocol!')

    i_inj = [0] * len(sweep_idxs)
    for i, sweep_idx in enumerate(sweep_idxs):
        if protocol == 'IV':
            amp_change = -0.15 + sweep_idx * 0.05
        elif protocol == 'rampIV':
            amp_change = sweep_idx * 0.1
        else:
            amp_change = 1
        i_inj[i] = list(i_inj_base * amp_change)
    return np.array(i_inj)


if __name__ == '__main__':
    cell_ids = ["2015_08_25b", "2015_08_25h", "2015_08_27d", "2015_08_26b", "2015_08_26f"]
    cell = '2015_08_26f'
    file_dir = os.path.join('/home/cf/Phd/DAP-Project/cell_data/raw_data', cell +'.dat')
    folder_name = 'vrest-80'
    v_rest = -80
    v_rest_shift = -16
    protocol = 'IV'

    v, t, sweep_idxs = get_v_and_t_from_heka(file_dir, protocol, group='Group1', trace='Trace1', sweep_idxs=None,
                                 return_sweep_idxs=True)
    i_inj = get_i_inj(protocol, sweep_idxs)  # re.sub('\(.*\)', '', protocol)
    v_set = set_v_rest(v, np.array([v[:, 0]]).T, np.ones((np.shape(v)[0], 1))*v_rest)
    v_shifted = shift_v_rest(v, v_rest_shift)

    for i in range(np.shape(v)[0]):
        fig, ax = pl.subplots(2, 1)
        ax[0].plot(t[i, :], v[i, :])
        #ax[0].plot(t[i, :], v_set[i, :])
        #ax[0].plot(t[i, :], v_shifted[i, :])
        ax[1].plot(t[i, :], i_inj[i, :])
        pl.show()
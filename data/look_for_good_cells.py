from data.hekareader import HekaReader
import matplotlib.pyplot as pl
import os
import pandas as pd
import numpy as np
from cell_characteristics.analyze_APs import get_v_rest, get_DAP_amp, get_AP_onsets, get_AP_max, get_fAHP_min, \
    get_DAP_max


def get_index_i_inj_start(i_inj):
    nonzero = np.nonzero(i_inj)[0]
    if len(nonzero) == 0:
        to_current = -1
    else:
        to_current = nonzero[0] - 1
    return to_current


if __name__ == '__main__':

    data_dir = '/home/caro/Downloads/rawData'
    for file_name in os.listdir(data_dir):
        hekareader = HekaReader(os.path.join(data_dir, file_name))
        type_to_index = hekareader.get_type_to_index()

        group = 'Group1'
        protocol_to_series = hekareader.get_protocol(group)

        if protocol_to_series.get('PP') is not None \
            and protocol_to_series.get('IV') is not None \
            and protocol_to_series.get('rampIV') is not None \
            and protocol_to_series.get('Zap20') is not None \
            and protocol_to_series.get('hypTester') is not None \
                and protocol_to_series.get('PP_tester') is not None:

            # check vrest does not change so much and noise in trace
            v_rest = list()
            v_noise = list()

            protocol = 'IV'
            trace = 'Trace1'
            series = protocol_to_series[protocol]
            sweeps = ['Sweep' + str(i) for i in range(1, len(type_to_index[group][series])+1)]
            sweep_idx = range(len(sweeps))
            sweeps = [sweeps[index] for index in sweep_idx][::3]
            indices = [type_to_index[group][series][sweep][trace] for sweep in sweeps]

            for i, index in enumerate(indices):
                # get voltage and time
                t, v = hekareader.get_xy(index)
                t *= 1000
                v *= 1000
                i_inj = pd.read_csv('./Protocols/' + protocol + '.csv', header=None)
                i_inj = np.array(i_inj)[:, 0]
                if protocol == 'IV':
                    amp = -0.15 + sweep_idx[i] * 0.05
                    amp_change = amp
                elif protocol == 'rampIV':
                    amp = sweep_idx[i] * 0.1
                    amp_change = amp / 0.1
                elif protocol == 'hypTester':
                    amp = -0.005
                    amp_change = 1
                elif protocol == 'Zap20':
                    amp = 0.1
                    amp_change = 1
                else:
                    amp = 0
                    amp_change = 1
                i_inj *= amp_change

                # check resting potential does not shift so much and noise
                v_rest.append(get_v_rest(v, i_inj))
                v_noise.append(np.var(v[0: get_index_i_inj_start(i_inj)]))

            diff_v_rest = np.max(v_rest) - np.min(v_rest)
            max_noise = np.max(v_noise)

            # check for DAP
            protocol = 'rampIV'
            trace = 'Trace1'
            series = protocol_to_series[protocol]
            sweeps = ['Sweep' + str(i) for i in range(1, len(type_to_index[group][series])+1)]
            sweep_idx = range(len(sweeps))
            sweeps = [[sweeps[index] for index in sweep_idx][-1]]
            indices = [type_to_index[group][series][sweep][trace] for sweep in sweeps]

            for i, index in enumerate(indices):
                # get voltage and time
                t, v = hekareader.get_xy(index)
                t *= 1000
                v *= 1000
                i_inj = pd.read_csv('./Protocols/' + protocol + '.csv', header=None)
                i_inj = np.array(i_inj)[:, 0]
                if protocol == 'IV':
                    amp = -0.15 + sweep_idx[i] * 0.05
                    amp_change = amp
                elif protocol == 'rampIV':
                    amp = sweep_idx[i] * 0.1
                    amp_change = amp / 0.1
                elif protocol == 'hypTester':
                    amp = -0.005
                    amp_change = 1
                elif protocol == 'Zap20':
                    amp = 0.1
                    amp_change = 1
                else:
                    amp = 0
                    amp_change = 1
                i_inj *= amp_change

                DAP_amp = 0
                vrest = get_v_rest(v, i_inj)
                AP_onsets = get_AP_onsets(v, threshold=-30)
                if len(AP_onsets) == 0:
                    continue
                AP_onset = AP_onsets[0]
                AP_end = -1
                dt = t[1]
                AP_max = get_AP_max(v, AP_onset, AP_end, interval=1 / dt)
                if AP_max is None:
                    continue
                fAHP_min = get_fAHP_min(v, AP_max, AP_end, interval=5 / dt)
                if fAHP_min is None:
                    continue
                DAP_max = get_DAP_max(v, fAHP_min, AP_end, interval=10 / dt)
                if DAP_max is None:
                    continue
                DAP_amp = get_DAP_amp(v, DAP_max, vrest)

            # select cells
            if DAP_amp > 10 and diff_v_rest <= 2 and max_noise <= 0.05:
                print file_name
                print 'diff v_rest: ', diff_v_rest
                print 'max noise: ', max_noise
                print 'DAP amp: ', DAP_amp
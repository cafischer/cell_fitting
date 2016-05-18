import numpy as np
import json
import pandas as pd

__author__ = 'caro'


def dicorator(dic, key):
    if key not in dic.keys():  # if the key does not exist yet it will be added with a new dict
        dic[key] = dict()
    return dic[key]

# TODO: imcomplete models can still be generated
def generate_model(variable_boundaries, save_dir=None):
    """

    :param variable_boundaries: list with entries:
        0: variable name
        1: lower boundary
        2: upper boundary
        3: path
    :type variable_boundaries: list
    :param save_dir: directory where model is saved
    :type save_dir: str
    :return: generated model with random parameters
    :rtype: dict
    """

    # choose random parameters
    variables_random = np.random.rand(len(variable_boundaries))
    variables_random = (variable_boundaries[:, 2] - variable_boundaries[:, 1]) * variables_random \
                       + variable_boundaries[:, 1]  # scale to real value according to boundaries

    # create model
    model = dict()
    for i, value in enumerate(variables_random):
        keys = variable_boundaries[i][3][0]
        reduce(dicorator, [model] + keys[:-1])[keys[-1]] = value

    # save model
    if save_dir is not None:
        with open(save_dir, 'w') as f:
            json.dump(model, f)

    return model, variables_random


def change_dt(dt, data, exp_protocol):
    times, amps, amp_types = from_protocol(exp_protocol)
    i = current_fromprotocol(times, amps, amp_types, dt)
    t = np.arange(0, times[-1]+dt, dt)

    data_new = pd.DataFrame()
    data_new.v = np.interp(t, data.t, data.v)
    data_new.t = t
    data_new.i = i
    data_new.sec = data.sec

    return data_new


def from_protocol(exp_protocol, stepamp=-0.05):
    if exp_protocol == 'step':
        times = [0, 500, 1500, 2000]
        amps = [0, stepamp, 0]
        amp_types = ['const', 'const', 'const']
    elif exp_protocol == 'ramp':
        times = [0, 10, 10.8, 12, 161.99]
        amps = [0, 0, 0.1, 0]
        amp_types = ['const', 'rampup', 'rampdown', 'const']
    elif exp_protocol == 'stepramp':
        times = [0, 200, 600, 602, 1002]
        amps = [0, stepamp, 6, 0]
        amp_types = ['const', 'const', 'ramp', 'const']
    else:
        print "Experimental Protocol not implemented, yet!"
    return times, amps, amp_types


def current_fromprotocol(times, amps, amp_types, dt):
    times_idx = np.array(times) / dt
    i = np.zeros(int(times[-1]/dt+1))

    for j in range(len(times_idx)-1):

        if amp_types[j] == 'const':
            i[int(times_idx[j]):int(times_idx[j+1])] = amps[j]
        elif amp_types[j] == 'ramp':
            if j == 0:
                amp_start = 0
            else:
                amp_start = amps[j-1]
            if j == len(times_idx)-1:
                amp_end = 0
            else:
                amp_end = amps[j+1]

            i[int(times_idx[j]):int(times_idx[j] + (times_idx[j+1]-times_idx[j])/2)] = np.linspace(amp_start, amps[j],
                                                                                        (times_idx[j+1]-times_idx[j])/2)
            i[int(times_idx[j] + (times_idx[j+1]-times_idx[j])/2):int(times_idx[j+1])] = np.linspace(amps[j], amp_end,
                                                                                        (times_idx[j+1]-times_idx[j])/2)
        elif amp_types[j] == 'rampup' or amp_types[j] == 'rampdown':
            i[int(times_idx[j]):int(times_idx[j+1])+1] = np.linspace(amps[j], amps[j+1],
                                                                     (times_idx[j+1]-times_idx[j])+1)
        else:
            raise ValueError('Unknown current amplitude type.')
    return i



if __name__ == "__main__":
    # TODO: write tests for replication of i_exp
    times, amps, amp_types = from_protocol('ramp')
    #i2 = current_fromprotocol(times, amps, amp_types, dt_exp)

    #pl.figure()
    #pl.plot(t_exp, i_exp)
    #pl.plot(t_exp, i2)
    #pl.show()
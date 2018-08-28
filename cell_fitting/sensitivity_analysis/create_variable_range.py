import os
import json
from nrn_wrapper import Cell, load_mechanism_dir
from cell_fitting.optimization.helpers import get_lowerbound_upperbound_keys
import numpy as np


def parameter_models_in_bounds(lower_bounds, upper_bounds, parameter_models):
    for parameter_model in parameter_models:
        if not np.all(lower_bounds < parameter_model) and np.all(parameter_model < upper_bounds):
            return False
    return True


def get_order_of_magnitude(x):
    return int(np.floor(np.log10(np.abs(x))))


if __name__ == '__main__':
    # parameters
    save_dir = os.path.join('../results/sensitivity_analysis/variable_ranges')
    variable_range_name = 'mean_std_1order_of_mag_model2'
    n_times_std = 1
    std_method = 'order_of_magnitude'

    model_dir = '../model/cells/dapmodel_simpel.json'
    mechanism_dir = '../model/channels/vavoulis'

    variables = [
        (np.nan, np.nan, [['soma', 'cm']]),
        (np.nan, np.nan, [['soma', '0.5', 'pas', 'e']]),
        (np.nan, np.nan, [['soma', '0.5', 'hcn_slow', 'ehcn']]),
        (np.nan, np.nan, [['soma', '0.5', 'pas', 'g']]),
        (np.nan, np.nan, [['soma', '0.5', 'nat', 'gbar']]),
        (np.nan, np.nan, [['soma', '0.5', 'nap', 'gbar']]),
        (np.nan, np.nan, [['soma', '0.5', 'kdr', 'gbar']]),
        (np.nan, np.nan, [['soma', '0.5', 'hcn_slow', 'gbar']]),
        (np.nan, np.nan, [['soma', '0.5', 'nat', 'm_vh']]),
        (np.nan, np.nan, [['soma', '0.5', 'nat', 'h_vh']]),
        (np.nan, np.nan, [['soma', '0.5', 'nap', 'm_vh']]),
        (np.nan, np.nan, [['soma', '0.5', 'nap', 'h_vh']]),
        (np.nan, np.nan, [['soma', '0.5', 'kdr', 'n_vh']]),
        (np.nan, np.nan, [['soma', '0.5', 'hcn_slow', 'n_vh']]),
        (np.nan, np.nan, [['soma', '0.5', 'nat', 'm_vs']]),
        (np.nan, np.nan, [['soma', '0.5', 'nat', 'h_vs']]),
        (np.nan, np.nan, [['soma', '0.5', 'nap', 'm_vs']]),
        (np.nan, np.nan, [['soma', '0.5', 'nap', 'h_vs']]),
        (np.nan, np.nan, [['soma', '0.5', 'kdr', 'n_vs']]),
        (np.nan, np.nan, [['soma', '0.5', 'hcn_slow', 'n_vs']]),
        (np.nan, np.nan, [['soma', '0.5', 'nat', 'm_tau_min']]),
        (np.nan, np.nan, [['soma', '0.5', 'nat', 'h_tau_min']]),
        (np.nan, np.nan, [['soma', '0.5', 'nap', 'm_tau_min']]),
        (np.nan, np.nan, [['soma', '0.5', 'nap', 'h_tau_min']]),
        (np.nan, np.nan, [['soma', '0.5', 'kdr', 'n_tau_min']]),
        (np.nan, np.nan, [['soma', '0.5', 'hcn_slow', 'n_tau_min']]),
        (np.nan, np.nan, [['soma', '0.5', 'nat', 'm_tau_max']]),
        (np.nan, np.nan, [['soma', '0.5', 'nat', 'h_tau_max']]),
        (np.nan, np.nan, [['soma', '0.5', 'nap', 'm_tau_max']]),
        (np.nan, np.nan, [['soma', '0.5', 'nap', 'h_tau_max']]),
        (np.nan, np.nan, [['soma', '0.5', 'kdr', 'n_tau_max']]),
        (np.nan, np.nan, [['soma', '0.5', 'hcn_slow', 'n_tau_max']]),
        (np.nan, np.nan, [['soma', '0.5', 'nat', 'm_tau_delta']]),
        (np.nan, np.nan, [['soma', '0.5', 'nat', 'h_tau_delta']]),
        (np.nan, np.nan, [['soma', '0.5', 'nap', 'm_tau_delta']]),
        (np.nan, np.nan, [['soma', '0.5', 'nap', 'h_tau_delta']]),
        (np.nan, np.nan, [['soma', '0.5', 'kdr', 'n_tau_delta']]),
        (np.nan, np.nan, [['soma', '0.5', 'hcn_slow', 'n_tau_delta']])
    ]

    # get values for parameters from the 6 models and use parameter ranges x stds around those
    model_dirs = ['/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/' + str(i) for i in
                  range(2, 3)] #range(1, 7)]
    model_dirs = [os.path.join(m_dir, 'cell.json') for m_dir in model_dirs]
    _, _, variable_keys = get_lowerbound_upperbound_keys(variables)
    parameter_models = np.zeros((len(model_dirs), len(variable_keys)))
    load_mechanism_dir(mechanism_dir)
    for m, m_dir in enumerate(model_dirs):
        cell = Cell.from_modeldir(m_dir)
        parameter_models[m, :] = [cell.get_attr(v_key[0]) for v_key in variable_keys]
    mean_parameter = np.mean(parameter_models, 0)
    #mean_parameter = parameter_models[2, :]

    if std_method == 'order_of_magnitude':
        order = np.array([get_order_of_magnitude(p) for p in mean_parameter])
        std_parameter = 10.**order
    elif std_method == 'std_over_models':
        std_parameter = np.std(parameter_models, 0)
    lower_bounds = mean_parameter - n_times_std * std_parameter
    upper_bounds = mean_parameter + n_times_std * std_parameter

    if not parameter_models_in_bounds(lower_bounds, upper_bounds, parameter_models):
        print 'Not all models in bounds!'

    variables = zip(lower_bounds, upper_bounds, variable_keys)
    for v in variables:
        print v

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, variable_range_name+'.json'), 'w') as f:
        json.dump(variables, f, indent=4)
import time
import os
import json
from cell_fitting.sensitivity_analysis import simulate_random_candidates


# parameters
save_dir = os.path.join('../results/sensitivity_analysis/', time.strftime('%Y-%m-%d_%H:%M:%S'))
n_candidates = int(1e5)
seed = time.time()

model_dir = '../model/cells/dapmodel_simpel.json'
mechanism_dir = '../model/channels/vavoulis'

data_dir = '../data/2015_08_26b/vrest-75/rampIV/3.0(nA).csv'
init_simulation_params = {'celsius': 35, 'onset': 200}  # must be dict (can be empty)

# variables = [
#             [0.3, 1, [['soma', 'cm']]],
#             [-90, -80, [['soma', '0.5', 'pas', 'e']]],
#             [-30, -10, [['soma', '0.5', 'hcn_slow', 'ehcn']]],
#
#             [0, 0.01, [['soma', '0.5', 'pas', 'g']]],
#             [0, 0.3, [['soma', '0.5', 'nat', 'gbar']]],
#             [0, 0.3, [['soma', '0.5', 'nap', 'gbar']]],
#             [0, 0.3, [['soma', '0.5', 'kdr', 'gbar']]],
#             [0, 0.001, [['soma', '0.5', 'hcn_slow', 'gbar']]],
#
#             [-90, -30, [['soma', '0.5', 'nat', 'm_vh']]],
#             [-90, -30, [['soma', '0.5', 'nat', 'h_vh']]],
#             [-90, -30, [['soma', '0.5', 'nap', 'm_vh']]],
#             [-90, -30, [['soma', '0.5', 'nap', 'h_vh']]],
#             [-90, -30, [['soma', '0.5', 'kdr', 'n_vh']]],
#             [-90, -30, [['soma', '0.5', 'hcn_slow', 'n_vh']]],
#
#             [10, 25, [['soma', '0.5', 'nat', 'm_vs']]],
#             [-25, -10, [['soma', '0.5', 'nat', 'h_vs']]],
#             [10, 25, [['soma', '0.5', 'nap', 'm_vs']]],
#             [-25, -10, [['soma', '0.5', 'nap', 'h_vs']]],
#             [10, 25, [['soma', '0.5', 'kdr', 'n_vs']]],
#             [-25, -10, [['soma', '0.5', 'hcn_slow', 'n_vs']]],
#
#             [0, 1, [['soma', '0.5', 'nat', 'm_tau_min']]],
#             [0, 1, [['soma', '0.5', 'nat', 'h_tau_min']]],
#             [0, 1, [['soma', '0.5', 'nap', 'm_tau_min']]],
#             [0, 1, [['soma', '0.5', 'nap', 'h_tau_min']]],
#             [0, 1, [['soma', '0.5', 'kdr', 'n_tau_min']]],
#             [0, 10, [['soma', '0.5', 'hcn_slow', 'n_tau_min']]],
#
#             [5, 30, [['soma', '0.5', 'nat', 'm_tau_max']]],
#             [5, 30, [['soma', '0.5', 'nat', 'h_tau_max']]],
#             [0, 1, [['soma', '0.5', 'nap', 'm_tau_max']]],
#             [0, 30, [['soma', '0.5', 'nap', 'h_tau_max']]],
#             [5, 30, [['soma', '0.5', 'kdr', 'n_tau_max']]],
#             [100, 150, [['soma', '0.5', 'hcn_slow', 'n_tau_max']]],
#
#             [0, 1, [['soma', '0.5', 'nat', 'm_tau_delta']]],
#             [0, 1, [['soma', '0.5', 'nat', 'h_tau_delta']]],
#             [0, 1, [['soma', '0.5', 'nap', 'm_tau_delta']]],
#             [0, 1, [['soma', '0.5', 'nap', 'h_tau_delta']]],
#             [0, 1, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
#             [0, 1, [['soma', '0.5', 'hcn_slow', 'n_tau_delta']]],
#             ]

variables = [
    (0.55286339468705603, 0.68087651424723439, [['soma', 'cm']]),
    (-88.057484033578504, -85.525730064284232, [['soma', '0.5', 'pas', 'e']]),
    (-30.867586111978, -22.839739108428468, [['soma', '0.5', 'hcn_slow', 'ehcn']]),
    (0.00037973518095184072, 0.001168949782115376, [['soma', '0.5', 'pas', 'g']]),
    (0.010580585856380404, 0.029306115932595137, [['soma', '0.5', 'nat', 'gbar']]),
    (0.11229303725896517, 0.22906742479787667, [['soma', '0.5', 'nap', 'gbar']]),
    (0.0028298968665886868, 0.010376680192710451, [['soma', '0.5', 'kdr', 'gbar']]),
    (5.5397296901621261e-05, 0.00012851180744634583, [['soma', '0.5', 'hcn_slow', 'gbar']]),
    (-55.38244233356356, -52.935541458340325, [['soma', '0.5', 'nat', 'm_vh']]),
    (-82.21768183066348, -79.347172457231551, [['soma', '0.5', 'nat', 'h_vh']]),
    (-33.925727713082964, -31.617593786204054, [['soma', '0.5', 'nap', 'm_vh']]),
    (-74.077571720483405, -61.83554303564221, [['soma', '0.5', 'nap', 'h_vh']]),
    (-67.870040886665777, -66.398494181563962, [['soma', '0.5', 'kdr', 'n_vh']]),
    (-84.36965629895758, -78.495323169754073, [['soma', '0.5', 'hcn_slow', 'n_vh']]),
    (15.322144157303617, 16.75608347138337, [['soma', '0.5', 'nat', 'm_vs']]),
    (-22.618180041089467, -19.128342570345893, [['soma', '0.5', 'nat', 'h_vs']]),
    (12.423700731228529, 14.505010240290808, [['soma', '0.5', 'nap', 'm_vs']]),
    (-14.259316896475026, -13.038143694527525, [['soma', '0.5', 'nap', 'h_vs']]),
    (18.090548747567762, 19.28706132401847, [['soma', '0.5', 'kdr', 'n_vs']]),
    (-20.559396057897217, -18.350561779406696, [['soma', '0.5', 'hcn_slow', 'n_vs']]),
    (-0.013573019863702795, 0.16930234030250638, [['soma', '0.5', 'nat', 'm_tau_min']]),
    (0.40043371497344277, 0.64123370831466897, [['soma', '0.5', 'nat', 'h_tau_min']]),
    (-0.0041495260488434507, 0.023248050553490964, [['soma', '0.5', 'nap', 'm_tau_min']]),
    (-0.002020779473301218, 0.18617446101935636, [['soma', '0.5', 'nap', 'h_tau_min']]),
    (0.19516538885767892, 0.52278071916767233, [['soma', '0.5', 'kdr', 'n_tau_min']]),
    (2.6019046957991208, 5.132609618672042, [['soma', '0.5', 'hcn_slow', 'n_tau_min']]),
    (16.129376570646983, 17.82584606592512, [['soma', '0.5', 'nat', 'm_tau_max']]),
    (14.226078984283934, 17.296034039623773, [['soma', '0.5', 'nat', 'h_tau_max']]),
    (0.066431429541228174, 0.20035235438134613, [['soma', '0.5', 'nap', 'm_tau_max']]),
    (8.150585388941483, 10.062896476447602, [['soma', '0.5', 'nap', 'h_tau_max']]),
    (21.219092061444531, 22.163422616935652, [['soma', '0.5', 'kdr', 'n_tau_max']]),
    (129.94787428208861, 137.24953060853977, [['soma', '0.5', 'hcn_slow', 'n_tau_max']]),
    (0.41479306443363217, 0.52551724803450373, [['soma', '0.5', 'nat', 'm_tau_delta']]),
    (0.37238869255622054, 0.77750185066043564, [['soma', '0.5', 'nat', 'h_tau_delta']]),
    (0.18515065273796882, 0.31980508853381218, [['soma', '0.5', 'nap', 'm_tau_delta']]),
    (0.23078746213017876, 0.43376568189095188, [['soma', '0.5', 'nap', 'h_tau_delta']]),
    (0.60921401149869781, 0.71078721763885089, [['soma', '0.5', 'kdr', 'n_tau_delta']]),
    (0.12104346090420127, 0.32860777830362142, [['soma', '0.5', 'hcn_slow', 'n_tau_delta']])
]

# # get values for parameters from the 6 models and use parameter ranges x stds around those
# from nrn_wrapper import Cell, load_mechanism_dir
# from cell_fitting.optimization.helpers import get_lowerbound_upperbound_keys
# import numpy as np
# model_dirs = ['/home/cf/Phd/programming/projects/cell_fitting/cell_fitting/results/best_models/'+str(i) for i in range(1,7)]
# model_dirs = [os.path.join(m_dir, 'cell.json') for m_dir in model_dirs]
# lower_bounds, upper_bounds, variable_keys = get_lowerbound_upperbound_keys(variables)
# parameter_models = np.zeros((len(model_dirs), len(variable_keys)))
# load_mechanism_dir(mechanism_dir)
# for m, m_dir in enumerate(model_dirs):
#     cell = Cell.from_modeldir(m_dir)
#     parameter_models[m, :] = [cell.get_attr(v_key[0]) for v_key in variable_keys]
# mean_parameter = np.mean(parameter_models, 0)
# std_parameter = np.std(parameter_models, 0)
#
# def parameter_models_in_bounds(lower_bounds, upper_bounds):
#     for parameter_model in parameter_models:
#         if not np.all(lower_bounds < parameter_model) and np.all(parameter_model < upper_bounds):
#             return False
#     return True
# i = 0
# while not parameter_models_in_bounds(lower_bounds, upper_bounds):
#     i += 1
#     lower_bounds = mean_parameter - i * std_parameter
#     upper_bounds = mean_parameter + i * std_parameter
# print 'Std around parameters: '+str(i)
# variables = zip(lower_bounds, upper_bounds, variable_keys)
# for v in variables:
#     print v

# create save_dir and save params
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

params = {'n_candidates': n_candidates, 'seed': seed, 'model_dir': model_dir, 'mechanism_dir': mechanism_dir,
          'variables': variables, 'data_dir': data_dir, 'init_simulation_params': init_simulation_params}

with open(os.path.join(save_dir, 'params.json'), 'w') as f:
    json.dump(params, f, indent=4)

# simulate
simulate_random_candidates(save_dir, **params)



import time
import os
import json
from cell_fitting.sensitivity_analysis import simulate_random_candidates


# parameters
save_dir = os.path.join('../results/sensitivity_analysis/', time.strftime('%Y-%m-%d_%H:%M:%S'))
n_candidates = int(500)
seed = time.time()

model_dir = '../model/cells/dapmodel_simpel.json'
mechanism_dir = '../model/channels/vavoulis'

data_dir = '../data/2015_08_26b/vrest-75/rampIV/3.0(nA).csv'
init_simulation_params = {'celsius': 35, 'onset': 200}  # must be dict (can be empty)

variables = [
            [0.3, 1, [['soma', 'cm']]],
            [-90, -80, [['soma', '0.5', 'pas', 'e']]],
            [-30, -10, [['soma', '0.5', 'hcn_slow', 'ehcn']]],

            [0, 0.01, [['soma', '0.5', 'pas', 'g']]],
            [0, 0.3, [['soma', '0.5', 'nat', 'gbar']]],
            [0, 0.3, [['soma', '0.5', 'nap', 'gbar']]],
            [0, 0.3, [['soma', '0.5', 'kdr', 'gbar']]],
            [0, 0.001, [['soma', '0.5', 'hcn_slow', 'gbar']]],

            [-90, -30, [['soma', '0.5', 'nat', 'm_vh']]],
            [-90, -30, [['soma', '0.5', 'nat', 'h_vh']]],
            [-90, -30, [['soma', '0.5', 'nap', 'm_vh']]],
            [-90, -30, [['soma', '0.5', 'nap', 'h_vh']]],
            [-90, -30, [['soma', '0.5', 'kdr', 'n_vh']]],
            [-90, -30, [['soma', '0.5', 'hcn_slow', 'n_vh']]],

            [10, 25, [['soma', '0.5', 'nat', 'm_vs']]],
            [-25, -10, [['soma', '0.5', 'nat', 'h_vs']]],
            [10, 25, [['soma', '0.5', 'nap', 'm_vs']]],
            [-25, -10, [['soma', '0.5', 'nap', 'h_vs']]],
            [10, 25, [['soma', '0.5', 'kdr', 'n_vs']]],
            [-25, -10, [['soma', '0.5', 'hcn_slow', 'n_vs']]],

            [0, 1, [['soma', '0.5', 'nat', 'm_tau_min']]],
            [0, 1, [['soma', '0.5', 'nat', 'h_tau_min']]],
            [0, 1, [['soma', '0.5', 'nap', 'm_tau_min']]],
            [0, 1, [['soma', '0.5', 'nap', 'h_tau_min']]],
            [0, 1, [['soma', '0.5', 'kdr', 'n_tau_min']]],
            [0, 10, [['soma', '0.5', 'hcn_slow', 'n_tau_min']]],

            [5, 30, [['soma', '0.5', 'nat', 'm_tau_max']]],
            [5, 30, [['soma', '0.5', 'nat', 'h_tau_max']]],
            [0, 1, [['soma', '0.5', 'nap', 'm_tau_max']]],
            [0, 30, [['soma', '0.5', 'nap', 'h_tau_max']]],
            [5, 30, [['soma', '0.5', 'kdr', 'n_tau_max']]],
            [100, 150, [['soma', '0.5', 'hcn_slow', 'n_tau_max']]],

            [0, 1, [['soma', '0.5', 'nat', 'm_tau_delta']]],
            [0, 1, [['soma', '0.5', 'nat', 'h_tau_delta']]],
            [0, 1, [['soma', '0.5', 'nap', 'm_tau_delta']]],
            [0, 1, [['soma', '0.5', 'nap', 'h_tau_delta']]],
            [0, 1, [['soma', '0.5', 'kdr', 'n_tau_delta']]],
            [0, 1, [['soma', '0.5', 'hcn_slow', 'n_tau_delta']]],
            ]

# create save_dir and save params
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

params = {'n_candidates': n_candidates, 'seed': seed, 'model_dir': model_dir, 'mechanism_dir': mechanism_dir,
          'variables': variables, 'data_dir': data_dir, 'init_simulation_params': init_simulation_params}

with open(os.path.join(save_dir, 'params.json'), 'w') as f:
    json.dump(params, f, indent=4)

# simulate
simulate_random_candidates(save_dir, **params)



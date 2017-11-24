import numpy as np
import time
import os
import json
from cell_fitting.sensitivity_analysis import simulate_random_candidates

# parameters
variable_range_name = 'mean_std_6models'  # 'mean3_std6models'
save_dir_params = os.path.join('../results/sensitivity_analysis/', variable_range_name)
save_dir = os.path.join('../results/sensitivity_analysis/', variable_range_name, time.strftime('%Y-%m-%d_%H:%M:%S'))
save_dir_range = os.path.join('../results/sensitivity_analysis/', 'variable_ranges')
n_candidates = int(100000)
seed = time.time()

model_dir = '../model/cells/dapmodel_simpel.json'
mechanism_dir = '../model/channels/vavoulis'

data_dir = '../data/cell_csv_data/2015_08_26b/rampIV/3.0(nA).csv'
init_simulation_params = {'celsius': 35, 'onset': 200, 'v_init': -75}  # must be dict (can be empty)
with open(os.path.join(save_dir_range, variable_range_name + '.json'), 'r') as f:
    variables = json.load(f)

# create save_dir and save params
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

params = {'n_candidates': n_candidates, 'seed': seed, 'model_dir': model_dir, 'mechanism_dir': mechanism_dir,
          'variables': variables, 'data_dir': data_dir, 'init_simulation_params': init_simulation_params}

with open(os.path.join(save_dir_params, 'params.json'), 'w') as f:
    json.dump(params, f, indent=4)

# simulate
simulate_random_candidates(save_dir, **params)



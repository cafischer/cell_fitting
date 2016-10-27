import os
import json
import numpy as np
import pandas as pd

from nrn_wrapper import *
from optimization.simulate import extract_simulation_params

__author__ = 'caro'

# parameter
save_dir = '../../../results/modellandscape/hhCell/zoom_gna_gk/'
p1_range = np.arange(0, 0.6, 0.0001)  # 0.12
p2_range = np.arange(0, 0.4, 0.0001)  # 0.036
variables = [
            [0, 1.0, [['soma', '0.5', 'na_hh', 'gnabar']]],
            [0, 1.0, [['soma', '0.5', 'k_hh', 'gkbar']]],
            #[0, 0.6, [['soma', '0.5', 'pas', 'g']]]
            ]
model_dir = '../../../model/cells/hhCell.json'
mechanism_dir = '../../../model/channels/hodgkinhuxley'
data_dir = '../../../data/toymodels/hhCell/ramp.csv'

# read data
data = pd.read_csv(data_dir)
simulation_params = extract_simulation_params(data, celsius=6.3)

# create cell
cell = Cell.from_modeldir(model_dir, mechanism_dir)

# compute error
modellandscape = np.zeros((len(p1_range), len(p2_range), len(data.t)))
for i, p1 in enumerate(p1_range):
    for j, p2 in enumerate(p2_range):

        # run simulation with these parameters
        cell.update_attr(variables[0][2][0], p1)
        cell.update_attr(variables[1][2][0], p2)
        v_model, t = iclamp(cell, **simulation_params)
        modellandscape[i, j, :] = v_model


# save models
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(save_dir+'/modellandscape.npy', 'w') as f:
    np.save(f, modellandscape)
np.savetxt(save_dir+'/p1_range.txt', p1_range)
np.savetxt(save_dir+'/p2_range.txt', p2_range)
with open(save_dir+'/variables.json', 'w') as f:
    json.dump(variables, f, indent=4)
with open(save_dir+'/dirs.json', 'w') as f:
    json.dump({'model_dir': model_dir, 'mechanism_dir': mechanism_dir, 'data_dir': data_dir}, f, indent=4)

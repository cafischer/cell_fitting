from __future__ import division
import os
import json
import numpy as np
import pandas as pd

from nrn_wrapper import *
from optimization.simulate import extract_simulation_params

__author__ = 'caro'

# parameter
save_dir = '../../results/fitnesslandscapes/modellandscape/gna_gka_highresolution/'
p1_range = np.arange(0, 1.0, 0.0001)  # 0.12
p2_range = np.arange(0, 1.0, 0.0001)  # 0.036
chunk_size = 100
variables = [
            [0, 1.0, [['soma', '0.5', 'na_hh', 'gnabar']]],
            [0, 1.0, [['soma', '0.5', 'k_hh', 'gkbar']]],
            #[0, 0.6, [['soma', '0.5', 'pas', 'g']]]
            ]
model_dir = '../../model/cells/hhCell.json'
mechanism_dir = '../../model/vclamp/hodgkinhuxley'
data_dir = '../../data/toymodels/hhCell/ramp.csv'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# read data
data = pd.read_csv(data_dir)
simulation_params = extract_simulation_params(data, celsius=6.3)

# create cell
cell = Cell.from_modeldir(model_dir, mechanism_dir)

# compute error
n_chunks_p1 = len(p1_range) / chunk_size
n_chunks_p2 = len(p2_range) / chunk_size
assert n_chunks_p1.is_integer()
assert n_chunks_p2.is_integer()
n_chunks_p1 = int(n_chunks_p1)
n_chunks_p2 = int(n_chunks_p2)
for c1 in range(n_chunks_p1):
    for c2 in range(n_chunks_p2):
        p1_chunk = p1_range[c1 * chunk_size:(c1 + 1) * chunk_size]
        p2_chunk = p2_range[c2 * chunk_size:(c2 + 1) * chunk_size]
        modellandscape = np.zeros((len(p1_chunk), len(p2_chunk), len(data.t)))
        for i, p1 in enumerate(p1_chunk):
            for j, p2 in enumerate(p2_chunk):

                # run simulation with these parameters
                cell.update_attr(variables[0][2][0], p1)
                cell.update_attr(variables[1][2][0], p2)
                v_model, t = iclamp(cell, **simulation_params)
                modellandscape[i, j, :] = v_model

        with open(save_dir + '/modellandscape'+str(c1)+'_'+str(c2)+ '.npy', 'w') as f:
            np.save(f, modellandscape)


# save models
with open(save_dir+'/chunk_size.txt', 'w') as f:
    f.write(str(chunk_size))
np.savetxt(save_dir+'/p1_range.txt', p1_range)
np.savetxt(save_dir+'/p2_range.txt', p2_range)
with open(save_dir+'/variables.json', 'w') as f:
    json.dump(variables, f, indent=4)
with open(save_dir+'/dirs.json', 'w') as f:
    json.dump({'model_dir': model_dir, 'mechanism_dir': mechanism_dir, 'data_dir': data_dir}, f, indent=4)

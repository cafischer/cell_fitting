from __future__ import division
import numpy as np
import pandas as pd
from optimization.fitnesslandscape import *

save_dir = '../../results/fitnesslandscapes/find_local_minima/combined_fitfuns/gna_gk/interpolate_v_trace+penalize_not1AP+APamp/L-BFGS-B/'
optimum = [0.12, 0.036]

candidates = pd.read_csv(save_dir + 'candidates.csv')
for index, row in candidates.iterrows():
    candidates.set_value(index, 'candidate', [float(x) for x in row.candidate.split(" ")])
n_candidates = len(candidates[candidates.generation == 0])

local_minimum_per_candidate = dict()
local_minimum_per_success = dict()

for index, row in candidates[candidates.generation == 0].iterrows():
    minimum = np.argmin(candidates[candidates.id == row.id].fitness)
    if minimum is np.nan:
        minimum = index
    if candidates.ix[np.argmax(candidates.generation[candidates.id == candidates.ix[minimum].id])].success == True:
        local_minimum_per_success[tuple(row.candidate)] = candidates.ix[minimum].candidate
    local_minimum_per_candidate[tuple(row.candidate)] = candidates.ix[minimum].candidate

distance = lambda x, y: np.abs(np.array(x) - np.array(y))
attraction_basins = assign_candidates_to_attraction_basin(local_minimum_per_candidate, distance, delta=10 ** (-3))
minima = attraction_basins.keys()

attraction_basins_success = assign_candidates_to_attraction_basin(local_minimum_per_success, distance, delta=10 ** (-3))
minima_success = attraction_basins_success.keys()

with open(save_dir + 'minima_descent.npy', 'w') as f:
    np.save(f, minima)

with open(save_dir + 'minima_success.npy', 'w') as f:
    np.save(f, minima_success)

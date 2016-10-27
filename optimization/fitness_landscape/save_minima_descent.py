from __future__ import division
import numpy as np
import pandas as pd
from optimization.fitness_landscape import *

save_dir = '../../results/fitness_landscape/find_local_minima/gna_gk/APamp/trust-ncg/'
optimum = [0.12, 0.036]

candidates = pd.read_csv(save_dir + 'candidates.csv')
for index, row in candidates.iterrows():
    candidates.set_value(index, 'candidate', [float(x) for x in row.candidate.split(" ")])
n_candidates = len(candidates[candidates.generation == 0])

local_minimum_per_candidate = dict()

for index, row in candidates[candidates.generation == 0].iterrows():
    minimum = np.argmin(candidates[candidates.id == row.id].fitness)
    if minimum is np.nan:
        minimum = index
    local_minimum_per_candidate[tuple(row.candidate)] = candidates.ix[minimum].candidate

distance = lambda x, y: np.abs(np.array(x) - np.array(y))
attraction_basins = assign_candidates_to_attraction_basin(local_minimum_per_candidate, distance, delta=10 ** (-3))
minima = attraction_basins.keys()

with open(save_dir + 'minima_descent.npy', 'w') as f:
    np.save(f, minima)


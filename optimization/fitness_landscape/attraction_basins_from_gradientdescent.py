from __future__ import division
import numpy as np
import pandas as pd
import itertools
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
n_attraction_basins = len(minima)
optimum_key = minima[np.argmin([np.sum(distance(optimum, minimum)) for minimum in minima])]
largest_key = minima[np.argmax([len(attraction_basins[minimum]) for minimum in minima])]
relative_size_optimum_attraction_basin = len(attraction_basins[optimum_key]) / n_candidates
relative_size_largest_attraction_basin = len(attraction_basins[largest_key]) / n_candidates
distance_optimum_largest = np.sum(distance(np.array(optimum_key), np.array(largest_key)))
mean_distance_minima = np.mean([np.sum(distance(np.array(comb[0]), np.array(comb[1])))
                                for comb in itertools.combinations(minima, 2)])


print 'Actual optimum: ' + str(optimum)
print 'Minimum of optimal attraction basin: ' + str(optimum_key)
print 'Minimum of largest attraction basin: ' + str(largest_key)
print 'Number of attraction basins: ' + str(n_attraction_basins)
print 'Relative size of optimal attraction basin: ' + str(relative_size_optimum_attraction_basin)
print 'Relative size of largest attraction basin: ' + str(relative_size_largest_attraction_basin)
print 'Distance between optimal and largest attraction basin: ' + str(distance_optimum_largest)
print 'Mean distance between basins: ' + str(mean_distance_minima)
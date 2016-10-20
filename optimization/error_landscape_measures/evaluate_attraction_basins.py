from __future__ import division
import numpy as np
import json
import itertools
from util import merge_dicts
from optimization.error_landscape_measures import *

save_dir = '../../results/fitness_landscape_analysis/hhCell/3params/rms/'

distance = lambda x, y: np.abs(np.array(x) - np.array(y))
def unmap(dictionary):
    return {tuple(pair['key']): pair['value'] for pair in dictionary}

# read
with open(save_dir + '/optimum.npy', 'r') as f:
    optimum = np.load(f)
with open(save_dir + '/n_candidates.txt', 'r') as f:
    n_candidates = int(f.read())
with open(save_dir + '/n_repeat.txt', 'r') as f:
    n_repeat = int(f.read())
n_repeat = 10

local_minimum_per_candidate = dict()
for i in range(n_repeat):
    with open(save_dir + 'local_minimum_per_candidate('+str(i)+').json', 'r') as f:
        local_minimum_per_candidate = merge_dicts(local_minimum_per_candidate, unmap(json.load(f)))

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
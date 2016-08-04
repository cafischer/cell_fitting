import pandas as pd
import numpy as np
import re

__author__ = 'caro'


methods = ['DEA', 'EDA', 'GA', 'SA', 'PSO']
save_dir = '../../results/bio_inspired/test_algorithms/increase_params/5param/'
n_trials = 10


best_fitness = pd.DataFrame(columns=methods, index=range(n_trials))
best_candidate = pd.DataFrame(columns=methods, index=range(n_trials))

for method in methods:
    for trial in range(n_trials):
        # read individuals_file
        path = save_dir+method+'/individuals_file_'+str(trial)+'.csv'
        individuals_file = pd.read_csv(path, dtype={'generation': np.int64, 'number': np.int64, 'fitness': np.float64,
                                                    'candidate': str})
        # find best
        n_generations = individuals_file.generation.iloc[-1]
        best = individuals_file.index[np.logical_and(individuals_file.generation.isin([n_generations]),
                                                     individuals_file.number.isin([0]))]
        best_fitness[method][trial] = individuals_file.fitness.iloc[best].values[0]
        best_candidate[method][trial] = individuals_file.candidate.iloc[best].values[0]

print best_fitness
print
print best_candidate


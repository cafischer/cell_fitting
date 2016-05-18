import pandas as pd
import numpy as np
import json

__author__ = 'caro'


method = 'DEA'
n_trials = 1000

best_candidates = list()
best_fitness = list()
method_args = list()

for trial in range(n_trials):
    # read in observed data from optimization
    path = './results_hyperparametersearch/'+method+'/individuals_file_'+str(trial)+'.csv'
    population_file = pd.read_csv(path, names=['generation', 'index', 'fitness', 'candidate'], header=None)
    with open('./results_hyperparametersearch/'+method+'/method_args_'+str(trial)+'.json', 'r') as f:
        method_args.append(json.load(f))

    n_generations = len(np.unique(population_file.generation))
    generation_file = population_file.loc[population_file.generation == n_generations-1]

    # find best candidate
    best = np.argmin(np.array(generation_file.fitness))
    best_candidates.append(generation_file.candidate.iloc[best])
    best_fitness.append(generation_file.fitness.iloc[best])

# find parameter set of best candidate over trials
# sort according to fitness
ranking = np.argsort(best_fitness)
best_fitness = np.array(best_fitness)[ranking]
method_args = np.array(method_args)[ranking]
print 'Fitness of best candidate: ' + str(best_fitness[0])
print 'Best hyperparameter set: ' + str(method_args[0])

n_best = 50
good_fitness = range(n_best)
print '\nMean and std of hyperparameter of best '+str(n_best)+' candidates: '
for key in method_args[0].keys():
    print key+': '
    print 'mean: ' + str(np.mean([method_args[i][key] for i in good_fitness])) \
          + '  std: ' + str(np.std([method_args[i][key] for i in good_fitness]))

threshold = 0.001
good_fitness = np.nonzero(best_fitness < threshold)[0]
print '\nNumber of candidates where error is lower than '+str(threshold)+': ' + str(len(good_fitness))
if np.size(good_fitness) > 1:
    print 'Mean and std of hyperparameter where error is lower than '+str(threshold)+': '
    for key in method_args[0].keys():
        print key+': '
        print 'mean: ' + str(np.mean([method_args[i][key] for i in good_fitness])) \
              + '  std: ' + str(np.std([method_args[i][key] for i in good_fitness]))

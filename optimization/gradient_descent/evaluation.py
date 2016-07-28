import pandas as pd
import numpy as np
import json
from neuron import h
from optimization.bio_inspired.problems import complete_mechanismdir
from optimization.bio_inspired.problems import CellFitProblem

__author__ = 'caro'


methods = ['Newton-CG', 'BFGS', 'Nelder-Mead', 'hhsolver']
save_dirs = ['../../results/gradient_descent/toymodels/toymodel1/1param/Newton-CG/',
             '../../results/gradient_descent/toymodels/toymodel1/1param/BFGS/',
            '../../results/gradient_descent/toymodels/toymodel1/1param/Nelder-Mead/',
            '../../results/hhsolver/toymodels/toymodel1/1param/higher_dt/']
n_trials = 10

with open(save_dirs[0]+'problem.json', 'r') as f:
    params = json.load(f)
h.nrn_load_dll(str(complete_mechanismdir(params['mechanism_dir'][3:])))  # TODO

best_fitness = pd.DataFrame(columns=methods, index=range(n_trials))
best_candidate = pd.DataFrame(columns=methods, index=range(n_trials))

for i, method in enumerate(methods):
    with open(save_dirs[i]+'problem.json', 'r') as f:
        params = json.load(f)
    params['data_dir'] = params['data_dir'][3:]  # TODO
    params['model_dir'] = params['model_dir'][3:]  # TODO
    params['mechanism_dir'] = None
    problem = CellFitProblem(**params)
    for trial in range(n_trials):
        candidate = [np.loadtxt(save_dirs[i]+'best_candidate_'+str(trial)+'.txt')]
        best_candidate[method][trial] = candidate

        best_fitness[method][trial] = problem.evaluate(candidate, None)

print best_fitness
print
print best_candidate


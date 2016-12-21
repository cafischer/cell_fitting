import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import json
from optimization.errfuns import rms
from new_optimization.fitter.hodgkinhuxleyfitter import HodgkinHuxleyFitter

__author__ = 'caro'


def plot_best_candidate(save_dir):
    candidates = pd.read_csv(save_dir + '/candidates.csv')
    best_candidate = candidates.candidate[np.argmin(candidates.fitness)]
    best_candidate = np.array([float(x) for x in best_candidate.split()])

    with open(save_dir + '/optimization_settings.json', 'r') as f:
        optimization_settings = json.load(f)

    fitter = HodgkinHuxleyFitter(**optimization_settings['fitter'])

    v_model, t, i_inj = fitter.simulate_cell(best_candidate)

    pl.figure()
    pl.plot(t, fitter.data.v, 'k', label='Data')
    pl.plot(t, v_model, 'r', label='Model')
    pl.legend(fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.ylabel('Membrane Potential (mV)', fontsize=16)
    pl.savefig(save_dir+'best_candidate.png')
    pl.show()

def plot_min_error_vs_generation(save_dir):

    candidates = pd.read_csv(save_dir + '/candidates.csv')

    best_fitnesses = list()

    for generation in range(candidates.generation.iloc[-1]+1):
        candidates_generation = candidates[candidates.generation == generation]
        best_fitnesses.append(candidates_generation.fitness[np.argmin(candidates_generation.fitness)])

    pl.figure()
    pl.plot(range(candidates.generation.iloc[-1]+1), best_fitnesses, 'k')
    pl.xlabel('Generation')
    pl.ylabel('Error')
    pl.savefig(save_dir+'error_development.png')
    pl.show()

if __name__ == '__main__':
    #save_dir = '../../results/new_optimization/2015_08_26b/test2/'
    save_dir = '../../results/fitnesslandscapes/follow_max_gradient/APamp+vrest+vtrace_withweighting20_withgpas/'
    method = 'L-BFGS-B'

    plot_best_candidate(save_dir+method+'/')
    plot_min_error_vs_generation(save_dir+method+'/')
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import json
from optimization.errfuns import rms
from new_optimization.fitter.hodgkinhuxleyfitter import HodgkinHuxleyFitter

__author__ = 'caro'


def plot_error_develoment(save_dir, id):

    candidates = pd.read_csv(save_dir + '/candidates.csv')
    candidate_data = candidates[candidates.id == id]
    n_generations = candidate_data.generation.iloc[-1]+1
    fitness_max_jac = list()
    fitnesses = np.zeros((3, n_generations))

    for generation in range(n_generations):
        candidate_generation = candidate_data[candidate_data.generation == generation]
        fitness = np.array([float(x) for x in candidate_generation.fitness.values[0].split()])
        idx_max_jac = candidate_generation.idx_max_jac.values
        fitnesses[:, generation] = fitness
        if not np.isnan(idx_max_jac):
            fitness_max_jac.append(fitnesses[int(idx_max_jac), generation-1])

    pl.figure()
    pl.plot(range(len(candidate_data)), fitnesses[0, :], 'b', label='APamp')
    pl.plot(range(len(candidate_data)), fitnesses[1, :], 'g', label='Vrest')
    pl.plot(range(len(candidate_data)), fitnesses[2, :], 'r', label='Vtrace')
    pl.plot(range(len(candidate_data)-1), fitness_max_jac[1:], 'ko', label='max gradient')
    pl.xlabel('Generation')
    pl.ylabel('Error')
    pl.legend()
    pl.savefig(save_dir+'error_development_candidate'+str(id)+'.png')
    pl.show()


def plot_movement_candidate(save_dir, id):
    candidates = pd.read_csv(save_dir + '/candidates.csv')
    candidate_data = candidates[candidates.id == id]
    n_generations = candidate_data.generation.iloc[-1]+1

    pl.figure()
    for generation in range(n_generations):
        candidate_generation = candidate_data[candidate_data.generation == generation]
        candidate_value = np.array([float(x) for x in candidate_generation.candidate.values[0].split()])
        pl.plot(candidate_value[0], candidate_value[1], 'xk', markeredgewidth=2)
        pl.text(candidate_value[0], candidate_value[1], str(generation), color='k', fontsize=12)
        pl.plot(0.12, 0.036, 'xr', markeredgewidth=2)

    pl.ylim(0, 0.4)
    pl.xlim(0, 0.5)
    pl.savefig(save_dir+'place_development_candidate'+str(id)+'.png')
    pl.show()


if __name__ == '__main__':
    save_dir = '../../results/fitnesslandscapes/follow_max_gradient/APamp+vrest+vtrace_withweighting100_withgpas/'
    method = 'L-BFGS-B'
    id = 4
    plot_error_develoment(save_dir+method+'/', id)
    plot_movement_candidate(save_dir+method+'/', id)
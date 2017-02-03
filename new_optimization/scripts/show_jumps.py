import matplotlib
#matplotlib.use('Agg')
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


def plot_trajectory_on_fitnesslandscapes(save_dir, save_dir_fitnesslandscapes, id):
    # get trajectory
    candidates = pd.read_csv(save_dir + '/candidates.csv')
    candidate_data = candidates[candidates.id == id]
    n_generations = candidate_data.generation.iloc[-1]+1

    trajectory = np.zeros((2, n_generations))
    for generation in range(n_generations):
        candidate_generation = candidate_data[candidate_data.generation == generation]
        candidate_value = np.array([float(x) for x in candidate_generation.candidate.values[0].split()])
        trajectory[:, generation] = [candidate_value[0], candidate_value[1]]

    # fitness landscape
    #with open(save_dir + '/optimization_settings.json', 'r') as f:
    #    optimization_settings = json.load(f)
    #fitfuns = optimization_settings['fitfun_names']
    fitfuns = ['APamp', 'v_rest', 'v_trace']
    optimum = [0.12, 0.036]
    p1_range = np.loadtxt(save_dir_fitnesslandscapes + '/p1_range.txt')
    p2_range = np.loadtxt(save_dir_fitnesslandscapes + '/p2_range.txt')

    for fitfun in fitfuns:
        with open(save_dir_fitnesslandscapes + '/fitfuns/' + fitfun + '/error.npy', 'r') as f:
            error = np.load(f)

        P1, P2 = np.meshgrid(p1_range, p2_range)
        fig, ax = pl.subplots()
        im = ax.pcolormesh(P1, P2, np.ma.masked_invalid(error).T)
        ax.plot(optimum[0], optimum[1], 'x', color='k', mew=2, ms=8)
        ax.plot(trajectory[0, :], trajectory[1, :], '-o', color='0.5', mew=2, ms=8)
        cmap = pl.get_cmap('gray')
        colors = [cmap(i) for i in np.linspace(0, 1, n_generations)]
        for i, color in enumerate(colors):
            pl.plot(trajectory[0, i], trajectory[1, i], '-o', color=color, mew=1, ms=6)
        pl.xlabel('$g_{na}$', fontsize=15)
        pl.ylabel('$g_{k}$', fontsize=15)
        pl.xlim(p1_range[0], p1_range[-1])
        pl.ylim(p2_range[0], p2_range[-1])
        pl.title(fitfun, fontsize=15)
        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel('Fitness', fontsize=15)
        pl.savefig(save_dir + fitfun + '_landscape_'+str(id)+'.png')
        pl.show()


if __name__ == '__main__':
    save_dir = '../../results/fitnesslandscapes/follow_max_gradient/APamp+vrest+vtrace_with0gradient/'
    method = 'L-BFGS-B'
    id = 4
    plot_error_develoment(save_dir+method+'/', id)
    plot_movement_candidate(save_dir+method+'/', id)

    #save_dir_fitnesslandscapes = '../../results/fitnesslandscapes/modellandscape/gna_gk_highresolution/'
    #plot_trajectory_on_fitnesslandscapes(save_dir+method+'/', save_dir_fitnesslandscapes, id)
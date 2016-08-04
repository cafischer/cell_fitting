import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as pl
from matplotlib import cycler
from pandas.tools.plotting import table
import os

from optimization.errfuns import rms
from optimization.problems import norm_candidate, get_variable_information, complete_mechanismdir
from nrn_wrapper import Cell
from neuron import h

__author__ = 'caro'


def get_candidate_fitness(save_dir, trial):
    save_dir_trial = save_dir + 'trial' + str(trial) + '/'

    # read individuals_file
    path = save_dir_trial+'individuals_file.csv'
    individuals_file = pd.read_csv(path, dtype={'generation': np.int64, 'number': np.int64, 'fitness': np.float64,
                                                'candidate': str})
    # find best candidate
    n_generations = individuals_file.generation.iloc[-1]
    best = individuals_file.index[np.logical_and(individuals_file.generation.isin([n_generations]),
                                                 individuals_file.number.isin([0]))]

    # convert string representation of candidate back to float
    candidate = individuals_file.candidate.iloc[best].values[0]
    candidate = re.sub('[\[\]]', '', candidate)
    candidate = np.array([float(x) for x in candidate.split()])

    # fitness
    fitness = individuals_file.fitness.iloc[best].values[0]

    return candidate, fitness

# ----------------------------------------------------------------------------------------------------------------------


def evaluate(methods, save_dir_statitics, save_dirs, n_trials, norm_weights):

    # load mechanisms
    with open(save_dirs[0]+'/specification/trial0/problem.json', 'r') as f:
        problem_dict = json.load(f)
    h.nrn_load_dll(str(complete_mechanismdir(problem_dict['mechanism_dir'])))  # TODO

    # create statistics dataframe
    columns = pd.MultiIndex(levels=[methods, ['rms(param)', 'rms(v)'], ['mean', 'var', 'min']],
                            labels=[list(np.repeat(range(len(methods)), 6)),
                                   [0, 0, 0, 1, 1, 1] * len(methods),
                                   [0, 1, 2, 0, 1, 2] * len(methods)],
                            names=['method', 'error', 'statistic'])
    index = pd.Index(data=range(1, len(save_dirs)+1), name='# params')
    statistics = pd.DataFrame(columns=columns, index=index)

    # get data
    for n, save_dir in enumerate(save_dirs, start=1):

        # initialize dataframes
        best_fitness = pd.DataFrame(columns=methods, index=range(n_trials))
        best_candidate = pd.DataFrame(columns=methods, index=range(n_trials))
        error_weights = pd.DataFrame(columns=methods, index=range(n_trials))

        for m, method in enumerate(methods):
            for trial in range(n_trials):

                # get fitness and parameter of best candidate
                candidate, fitness = get_candidate_fitness(save_dir+'/'+method+'/', trial)
                best_candidate[method][trial] = candidate
                best_fitness[method][trial] = fitness

                # get optimal candidate
                with open(save_dir + 'specification/trial' + str(trial) + '/problem.json', 'r') as f:
                    problem_dict = json.load(f)
                cell = Cell.from_modeldir(problem_dict['model_dir'])
                optimal_candidate = [cell.get_attr(var[2][0]) for var in problem_dict['variables']]

                # computer error in parameter
                if norm_weights:
                    lower_bound, upper_bound, path_variables = get_variable_information(problem_dict['variables'])
                    optimal_candidate_normed = norm_candidate(optimal_candidate, lower_bound, upper_bound)
                    candidate_normed = norm_candidate(candidate, lower_bound, upper_bound)
                    error_weights[method][trial] = np.mean(np.array([rms(candidate_normed[k], optimal_candidate_normed[k])
                                                                     for k in range(len(optimal_candidate))]))
                else:
                    error_weights[method][trial] = np.mean(np.array([rms(candidate[k], optimal_candidate[k])
                                                                 for k in range(len(optimal_candidate))]))

        # statistics
        statistics.loc[n, (slice(None), 'rms(param)', 'mean')] = np.mean(error_weights, 0).values
        statistics.loc[n, (slice(None), 'rms(param)', 'var')] = np.var(error_weights, 0).values
        statistics.loc[n, (slice(None), 'rms(param)', 'min')] = np.min(error_weights, 0).values
        statistics.loc[n, (slice(None), 'rms(v)', 'mean')] = np.mean(best_fitness, 0).values
        statistics.loc[n, (slice(None), 'rms(v)', 'var')] = np.var(best_fitness, 0).values
        statistics.loc[n, (slice(None), 'rms(v)', 'min')] = np.min(best_fitness, 0).values

    print statistics
    if not os.path.exists(save_dir_statistics):
        os.makedirs(save_dir_statistics)
    statistics.to_csv(save_dir_statistics+'statistics.csv')


def plot_data(save_dir_statistics, methods, errors):
    statistics = pd.read_csv(save_dir_statistics+'statistics.csv', header=[0, 1, 2], index_col=[0])
    statistics.sortlevel(axis=0, inplace=True, sort_remaining=True)
    statistics.sortlevel(axis=1, inplace=True, sort_remaining=True)
    statistics[statistics == np.inf] = np.nan
    print statistics

    # change color cycle
    cmap = pl.get_cmap('jet')
    colors = cmap(np.linspace(0.1, 0.9, len(methods)))

    # index for sorting dataframe methods
    idx = np.argsort(np.argsort(methods))  # second argsort do get idx for undoing sorting

    for error in errors:

        # plot minimum
        statistics_min = statistics.loc[(slice(None)), (slice(None), error, 'min')]

        fig = pl.figure(figsize=(9, 8))
        ax = fig.add_subplot(111)
        ax.set_prop_cycle(cycler('color', colors))
        for method in methods:
            ax.plot(statistics.index, statistics_min[method], label=method)
        table(ax, statistics_min[idx].transpose().apply(lambda x: x.map(lambda y: "%.6f" % y)),
              rowLabels=methods, loc='bottom',  bbox=[0, -0.7, 1, 0.55])
        pl.tight_layout(rect=[0.15, 0.36, 1.0, 1.0])
        ax.set_xticks(statistics.index)
        ax.set_xticklabels(statistics.index)

        pl.legend(fontsize=12)
        pl.ylim([0, None])
        pl.xlabel(statistics.index.name)
        pl.ylabel('min '+error)
        pl.savefig(save_dir_statistics+'/plot_'+error+'_min.png')
        pl.show()

        # plot mean
        statistics_mean = statistics.loc[(slice(None)), (slice(None), error, 'mean')]
        statistics_std = np.sqrt(statistics.loc[(slice(None)), (slice(None), error, 'var')])

        fig = pl.figure(figsize=(9, 8))
        ax = fig.add_subplot(111)
        ax.set_prop_cycle(cycler('color', colors))
        for method in methods:
            base_line, = ax.plot(statistics.index, statistics_mean[method], label=method)

            ax.fill_between(statistics.index.values,
                            (statistics_mean[method].values - statistics_std[method].values).flatten(),
                            (statistics_mean[method].values + statistics_std[method].values).flatten(),
                            facecolor=base_line.get_color(),  alpha=0.1)
        table(ax, statistics_mean[idx].transpose().apply(lambda x: x.map(lambda y: "%.6f" % y)),
              rowLabels=methods, loc='bottom',  bbox=[0, -0.7, 1, 0.55])
        pl.tight_layout(rect=[0.15, 0.36, 1.0, 1.0])
        ax.set_xticks(statistics.index)
        ax.set_xticklabels(statistics.index)

        pl.legend(fontsize=12)
        pl.xlabel(statistics.index.name)
        pl.ylim([0, None])
        pl.ylabel('mean '+error)
        pl.savefig(save_dir_statistics+'/plot_'+error+'_mean.png')
        pl.show()

# ---------------------------------------------------------------------------------------------------------------------

# parameter
methods = ['DEA', 'SA', 'PSO', 'L-BFGS-B', 'Nelder-Mead','random']
method_types = ['ec', 'ec', 'swarm', 'gradient_based', 'simplex', 'random']
save_dir_statistics = '../results/test/statistic/'
save_dirs = ['../../results/test/1param/',
             '../../results/test/2param/'
             ]
n_trials = 2

norm_weights = True


evaluate(methods, save_dir_statistics, save_dirs, n_trials, norm_weights)
errors = ['rms(v)', 'rms(param)']
plot_data(save_dir_statistics, methods, errors)

# TODO: maybe cut unbounded methods to 1 in normalize condition
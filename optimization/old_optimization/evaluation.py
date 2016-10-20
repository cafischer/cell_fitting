from __future__ import division

import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as pl
from matplotlib import cycler
from pandas.tools.plotting import table
import os
from itertools import combinations
import numpy.ma as ma

from optimization.errfuns import rms
from optimization.helpers import get_lowerbound_upperbound_keys
from nrn_wrapper import Cell, load_mechanism_dir

__author__ = 'caro'


class Evaluator:

    def __init__(self, save_dir_statistics, save_dirs, n_trials, methods):
        self.save_dir_statistics = save_dir_statistics
        self.save_dirs = save_dirs
        self.n_trials = n_trials
        self.methods = methods

    def save_statistics(self, name_error_weights='error_weights', name_best_fitness='best_fitness'):
        statistics = self.create_statistics_dataframe()

        for n, save_dir in enumerate(self.save_dirs, start=1):
            error_weights = pd.read_csv(save_dir+name_error_weights+'.csv', index_col=0)
            best_fitness = pd.read_csv(save_dir+name_best_fitness+'.csv', index_col=0)
            statistics.loc[n, (slice(None), 'rms(param)', 'mean')] = np.mean(error_weights, 0).values
            statistics.loc[n, (slice(None), 'rms(param)', 'var')] = np.var(error_weights, 0).values
            statistics.loc[n, (slice(None), 'rms(param)', 'min')] = np.min(error_weights, 0).values
            statistics.loc[n, (slice(None), 'rms(v)', 'mean')] = np.mean(best_fitness, 0).values
            statistics.loc[n, (slice(None), 'rms(v)', 'var')] = np.var(best_fitness, 0).values
            statistics.loc[n, (slice(None), 'rms(v)', 'min')] = np.min(best_fitness, 0).values

        if not os.path.exists(self.save_dir_statistics):
            os.makedirs(self.save_dir_statistics)
        statistics.to_csv(self.save_dir_statistics+'statistics.csv')

    def create_statistics_dataframe(self):
        columns = pd.MultiIndex(levels=[self.methods, ['rms(param)', 'rms(v)'], ['mean', 'var', 'min']],
                                labels=[list(np.repeat(range(len(self.methods)), 6)),
                                        [0, 0, 0, 1, 1, 1] * len(self.methods),
                                        [0, 1, 2, 0, 1, 2] * len(self.methods)],
                                names=['method', 'error', 'statistic'])
        index = pd.Index(data=range(1, len(self.save_dirs) + 1), name='# params')
        statistics = pd.DataFrame(columns=columns, index=index)
        return statistics

    def save_error_weights_and_best_fitness(self, norm_weights=True, selected_trials=None,
                                            name_error_weights='error_weights', name_best_fitness='best_fitness'):

        self.load_mechanisms()

        for n, save_dir in enumerate(self.save_dirs):

            best_candidate = pd.DataFrame(columns=self.methods, index=range(self.n_trials))
            best_fitness = pd.DataFrame(columns=self.methods, index=range(self.n_trials))
            error_weights = pd.DataFrame(columns=self.methods, index=range(self.n_trials))

            for m, method in enumerate(self.methods):
                if selected_trials is None:
                    trials = range(self.n_trials)
                else:
                    trials = selected_trials[n]

                for trial in trials:

                    candidate, fitness = self.get_candidate_and_fitness(save_dir+'/'+method+'/', trial)
                    best_candidate[method][trial] = candidate
                    best_fitness[method][trial] = fitness
                    problem_dict = self.get_problem_dict(save_dir, trial)
                    optimal_candidate = self.get_optimal_candidate(problem_dict['model_dir'], problem_dict['variables'])

                    if norm_weights:
                        error_weights[method][trial] = self.get_mean_rms_candidate_normed(candidate, optimal_candidate,
                                                                                          problem_dict['variables'],
                                                                                          method)
                        assert 0.0 <= error_weights[method][trial] <= 1.0
                    else:
                        error_weights[method][trial] = self.get_mean_rms_candidate_variables(candidate, optimal_candidate)

            error_weights.to_csv(save_dir+name_error_weights+'.csv')
            best_fitness.to_csv(save_dir+name_best_fitness+'.csv')

    def get_mean_rms_candidate_variables(self, candidate, optimal_candidate):
        return np.mean(np.array([rms(candidate[k], optimal_candidate[k])
                                 for k in range(len(optimal_candidate))]))

    def get_mean_rms_candidate_normed(self, candidate, optimal_candidate, variables, method):
        lower_bound, upper_bound, path_variables = get_lowerbound_upperbound_keys(variables)
        optimal_candidate_normed = NormalizedProblem.norm_candidate(optimal_candidate, lower_bound, upper_bound)
        candidate_normed = NormalizedProblem.norm_candidate(candidate, lower_bound, upper_bound)
        if method == 'Nelder-Mead':
            candidate_normed = self.norm_candidate_unbounded_method(candidate_normed)
        rms_candidate_normed = self.get_mean_rms_candidate_variables(candidate_normed, optimal_candidate_normed)
        return rms_candidate_normed

    def norm_candidate_unbounded_method(self, candidate_normed):
        candidate_normed = [1.0 if x > 1.0 else x for x in candidate_normed]  # for unbounded methods cut at 1
        candidate_normed = [0.0 if x < 0.0 else x for x in candidate_normed]
        return candidate_normed

    def load_mechanisms(self):
        with open(self.save_dirs[0] + '/specification/trial0/problem.json', 'r') as f:
            problem_dict = json.load(f)
        load_mechanism_dir(str(problem_dict['mechanism_dir']))

    def get_optimal_candidate(self, model_dir, variables):
        cell = Cell.from_modeldir(model_dir)
        optimal_candidate = [cell.get_attr(var[2][0]) for var in variables]
        return optimal_candidate

    def get_problem_dict(self, save_dir, trial):
        with open(save_dir + 'specification/trial' + str(trial) + '/problem.json', 'r') as f:
            problem_dict = json.load(f)
        return problem_dict

    def plot_statistic(self, error, statistic='mean'):
        statistics = pd.read_csv(self.save_dir_statistics+'statistics.csv', header=[0, 1, 2], index_col=[0])
        statistics.sortlevel(axis=0, inplace=True, sort_remaining=True)
        statistics.sortlevel(axis=1, inplace=True, sort_remaining=True)
        statistics[statistics == np.inf] = np.nan

        # index for sorting dataframe methods
        idx = np.argsort(np.argsort(self.methods))  # second argsort do get idx for undoing sorting

        # change color cycle
        cmap = pl.get_cmap('jet')
        colors = cmap(np.linspace(0.1, 0.9, len(self.methods)))

        fig = pl.figure(figsize=(9, 8))
        ax = fig.add_subplot(111)
        ax.set_prop_cycle(cycler('color', colors))
        for method in self.methods:
            if statistic == 'min':
                statistic_min = statistics.loc[(slice(None)), (slice(None), error, statistic)]
                ax.plot(statistics.index, statistic_min[method], label=method)
                table(ax, statistic_min[idx].transpose().apply(lambda x: x.map(lambda y: "%.6f" % y)),
                  rowLabels=self.methods, loc='bottom', bbox=[0, -0.7, 1, 0.55])
            elif statistic == 'mean':
                statistic_mean = statistics.loc[(slice(None)), (slice(None), error, statistic)]
                statistic_std = np.sqrt(statistics.loc[(slice(None)), (slice(None), error, 'var')])
                base_line, = ax.plot(statistics.index, statistic_mean[method], label=method)

                ax.fill_between(statistics.index.values,
                                (statistic_mean[method].values - statistic_std[method].values).flatten(),
                                (statistic_mean[method].values + statistic_std[method].values).flatten(),
                                facecolor=base_line.get_color(), alpha=0.1)
                table(ax, statistic_mean[idx].transpose().apply(lambda x: x.map(lambda y: "%.6f" % y)),
                  rowLabels=self.methods, loc='bottom', bbox=[0, -0.7, 1, 0.55])
        pl.tight_layout(rect=[0.15, 0.36, 1.0, 1.0])
        ax.set_xticks(statistics.index)
        ax.set_xticklabels(statistics.index)
        pl.legend(fontsize=12)
        pl.ylim([0, None])
        pl.xlabel(statistics.index.name)
        pl.ylabel(statistic + ' ' + error)
        pl.savefig(self.save_dir_statistics+'/plot_'+error+'_'+statistic+'.png')
        pl.show()

    def get_candidate_and_fitness(self, save_dir, trial):
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

    def hist_mean_error_variable_comb(self, save_dir, n_param, name_error_weights='error_weights'):

        if not os.path.exists(self.save_dir_statistics+'histograms/'+str(n_param)+'param/'):
            os.makedirs(self.save_dir_statistics+'histograms/'+str(n_param)+'param/')

        error_weights = pd.read_csv(save_dir+name_error_weights+'.csv')

        dict_variables, variable_names, range_variables = self.get_dict_variables(save_dir)
        dict_variables_inv = {v: k for k, v in dict_variables.items()}
        variable_combinations = list(combinations(range_variables, n_param))
        variable_combinations_names = [tuple([dict_variables_inv[e] for e in comb]) for comb in variable_combinations]

        for m, method in enumerate(self.methods):
            variable_comb_mean_error_dict = self.get_mean_error_foreach_variable_combination(
                error_weights[method].values,
                variable_combinations,
                save_dir, dict_variables)
            variable_comb_mean_error = np.array([variable_comb_mean_error_dict[vc] for vc in variable_combinations])

            w = 0.8
            pl.bar(range(len(variable_combinations)), variable_comb_mean_error, width=w, color='k', align='center',
                   label='mean rms(param)')
            pl.xticks(range(len(variable_combinations)), variable_combinations_names, rotation='vertical')
            pl.title(method)
            pl.ylim(0.0, None)
            pl.legend()
            pl.tight_layout()
            pl.savefig(self.save_dir_statistics + 'histograms/'+str(n_param)+'param/'+method+'.png')
            pl.show()

    def plot_2d_mean_error_variable_comb(self, name_error_weights='error_weights'):

        save_dir = self.save_dirs[1]  # for 2 parameters

        if not os.path.exists(self.save_dir_statistics+'2dplot/'):
            os.makedirs(self.save_dir_statistics+'2dplot/')

        error_weights = pd.read_csv(save_dir+name_error_weights+'.csv')

        dict_variables, variable_names, range_variables = self.get_dict_variables(save_dir)
        variable_combinations = list(combinations(range_variables, 2))

        for method in self.methods:
            variable_comb_mean_error_dict = self.get_mean_error_foreach_variable_combination(error_weights[method].values,
                                                                                           variable_combinations,
                                                                                           save_dir, dict_variables)

            x_2d, y_2d = np.meshgrid(range_variables, range_variables)
            z_2d = np.zeros((len(range_variables), len(range_variables)))
            z_2d[:] = np.nan
            for variableComb, error in variable_comb_mean_error_dict.iteritems():
                z_2d[variableComb] = error

            z_masked = ma.masked_where(np.isnan(z_2d), z_2d)

            pl.figure()
            pl.pcolormesh(x_2d, y_2d, z_masked, vmin=0.0, vmax=1.0)
            pl.gca().set_aspect("equal")
            cbar = pl.colorbar()
            cbar.set_label('mean rms(param)')
            pl.title(method)
            pl.xlim(0, len(range_variables))
            pl.ylim(0, len(range_variables))
            pl.xticks(np.array(range_variables)+0.5, variable_names, rotation='vertical')
            pl.yticks(np.array(range_variables)+0.5, variable_names)
            pl.tight_layout()
            pl.savefig(self.save_dir_statistics+'2dplot/'+method+'.png')
            pl.show()

    def get_dict_variables(self, save_dir):
        with open(save_dir + 'specification/variables_all.json', 'r') as f:
            all_variables = json.load(f)
        all_names = [all_variables[i][2][0][-1] for i in range(len(all_variables))]
        range_names = range(len(all_names))
        dict_names = dict((all_names[i], range_names[i]) for i in range(len(all_names)))

        return dict_names, all_names, range_names

    def get_mean_error_foreach_variable_combination(self, error_weights_of_method, variable_combinations,
                                                    save_dir, dict_variables):
        variable_comb_error_trials_dict = {k: list() for k in variable_combinations}
        variable_comb_mean_error_dict = dict()

        for trial in range(self.n_trials):
            variable_comb_per_trial = self.get_translated_variable_comb(save_dir, trial, dict_variables)
            variable_comb_error_trials_dict[variable_comb_per_trial].append(error_weights_of_method[trial])

        for variable_comb in variable_combinations:
            variable_comb_mean_error_dict[variable_comb] = np.mean(variable_comb_error_trials_dict[variable_comb])

        return variable_comb_mean_error_dict

    def get_translated_variable_comb(self, save_dir, trial, dict_variables):
        with open(save_dir + 'specification/trial' + str(trial) + '/problem.json', 'r') as f:
            problem_dict = json.load(f)
        translated_variable_comb = [dict_variables[problem_dict['variables'][i][2][0][-1]]
                                      for i in range(len(problem_dict['variables']))]
        translated_variable_comb.sort()
        return tuple(translated_variable_comb)


if __name__ == '__main__':
    # parameter
    methods = ['DEA', 'SA', 'PSO', 'L-BFGS-B', 'Nelder-Mead', 'random']
    method_types = ['ec', 'ec', 'swarm', 'gradient_based', 'simplex', 'random']
    save_dir_statistics = '../../results/algorithms_on_hhcell/statistic/'
    save_dirs = ['../../results/algorithms_on_hhcell/1param/',
                 '../../results/algorithms_on_hhcell/2param/',
                 '../../results/algorithms_on_hhcell/3param/',
                 '../../results/algorithms_on_hhcell/4param/',
                 '../../results/algorithms_on_hhcell/5param/',
                 '../../results/algorithms_on_hhcell/6param/',
                 '../../results/algorithms_on_hhcell/7param/',
                 '../../results/algorithms_on_hhcell/8param/',
                 '../../results/algorithms_on_hhcell/9param/',
                 '../../results/algorithms_on_hhcell/10param/'
                 ]
    n_trials = 100
    norm_weights = True

    evaluator = Evaluator(save_dir_statistics, save_dirs, n_trials, methods)
    #evaluator.save_error_weights_and_best_fitness()
    evaluator.save_statistics()
    evaluator.plot_statistic('rms(param)', 'mean')
    evaluator.plot_statistic('rms(param)', 'min')
    evaluator.plot_statistic('rms(v)', 'mean')
    evaluator.plot_statistic('rms(v)', 'min')
    evaluator.hist_mean_error_variable_comb(save_dirs[0], 1)
    evaluator.plot_2d_mean_error_variable_comb()
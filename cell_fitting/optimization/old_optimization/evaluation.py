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
from cell_fitting.optimization.simulate import extract_simulation_params
from cell_fitting.optimization.errfuns import rms
from cell_fitting.optimization.helpers import get_lowerbound_upperbound_keys
from nrn_wrapper import *
pl.style.use('paper')
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
        statistics.to_csv(self.save_dir_statistics+'cell_characteristics.csv')

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
        statistics = pd.read_csv(self.save_dir_statistics+'cell_characteristics.csv', header=[0, 1, 2], index_col=[0])
        statistics.sortlevel(axis=0, inplace=True, sort_remaining=True)
        statistics.sortlevel(axis=1, inplace=True, sort_remaining=True)
        statistics[statistics == np.inf] = np.nan

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
                table(ax, statistic_min.transpose().apply(lambda x: x.map(lambda y: "%.6f" % y)),
                  rowLabels=statistic_mean.columns.levels[0].values, loc='bottom', bbox=[0, -0.7, 1, 0.55])
            elif statistic == 'mean':
                statistic_mean = statistics.loc[(slice(None)), (slice(None), error, statistic)]
                statistic_std = np.sqrt(statistics.loc[(slice(None)), (slice(None), error, 'var')])
                base_line, = ax.plot(statistics.index, statistic_mean[method], label=method, linewidth=1.5)

                ax.fill_between(statistics.index.values,
                                (statistic_mean[method].values - statistic_std[method].values).flatten(),
                                (statistic_mean[method].values + statistic_std[method].values).flatten(),
                                facecolor=base_line.get_color(), alpha=0.05)
                table(ax, statistic_mean.transpose().apply(lambda x: x.map(lambda y: "%.6f" % y)),
                  rowLabels=statistic_mean.columns.levels[0].values, loc='bottom', bbox=[0, -0.7, 1, 0.55])
        pl.tight_layout(rect=[0.15, 0.36, 1.0, 1.0])
        ax.set_xticks(statistics.index)
        ax.set_xticklabels(statistics.index)
        pl.legend(fontsize=14)
        pl.ylim([0, None])
        #pl.xlabel(statistics.index.name, fontsize=16)
        pl.xlabel('$\# param$', fontsize=18)
        #pl.ylabel(statistic + '(' + error + ')', fontsize=16)
        #pl.ylabel('$mean_{trials}[rms(param_{to fit}, param_{best})]$', fontsize=18)
        pl.ylabel('$mean_{trials}[rms(V_{to fit}, V_{best})]$', fontsize=18)
        pl.savefig(self.save_dir_statistics+'/plot_'+error+'_'+statistic+'.png')
        pl.show()

    def plot_statistic_without_table(self, errors=None, statistic='mean'):
        errors = ['rms(param)', 'rms(v)'] if errors is None else errors
        statistics = pd.read_csv(self.save_dir_statistics+'cell_characteristics.csv', header=[0, 1, 2], index_col=[0])
        statistics.sortlevel(axis=0, inplace=True, sort_remaining=True)
        statistics.sortlevel(axis=1, inplace=True, sort_remaining=True)
        statistics[statistics == np.inf] = np.nan

        # change color cycle
        # cmap = pl.get_cmap('Set2')  # jet
        # colors = cmap(np.linspace(0.1, 0.9, len(self.methods)))
        colors = ['xkcd:red', 'xkcd:orange', 'm', 'y', 'b', 'grey']

        fig, axes = pl.subplots(1, 2, figsize=(11, 5))
        for error_idx, error in enumerate(errors):
            ax = axes[error_idx]
            ax.set_prop_cycle(cycler('color', colors))
            for method in self.methods:
                if statistic == 'min':
                    statistic_min = statistics.loc[(slice(None)), (slice(None), error, statistic)]
                    ax.plot(statistics.index, statistic_min[method], label=method)
                elif statistic == 'mean':
                    statistic_mean = statistics.loc[(slice(None)), (slice(None), error, statistic)]
                    statistic_std = np.sqrt(statistics.loc[(slice(None)), (slice(None), error, 'var')])
                    base_line, = ax.plot(statistics.index, statistic_mean[method], label=method, linewidth=1.5)

                    # TODO
                    n_samples = 100
                    ax.fill_between(statistics.index.values,
                                    (statistic_mean[method].values - statistic_std[method].values / np.sqrt(n_samples)).flatten(),
                                    (statistic_mean[method].values + statistic_std[method].values / np.sqrt(n_samples)).flatten(),
                                    facecolor=base_line.get_color(), alpha=0.1)
                ax.set_xticks(statistics.index)
                ax.set_xticklabels(statistics.index)
                ax.set_xlabel('# Parameter')
                ax.set_xlim(1, len(statistic_mean[method].values))
                if error == 'rms(v)':
                    ax.set_ylabel('RMSE mem. pot. (mV)')
                    ax.set_ylim([0, 16.5])
                elif error == 'rms(param)':
                    ax.set_ylabel('RMSE parameters (norm.)')
                    ax.set_ylim([0, 0.35])
        axes[0].legend(loc='upper left')
        pl.tight_layout()
        pl.savefig(os.path.join(self.save_dir_statistics, 'plot_rmse_'+statistic+'.png'))
        pl.savefig(os.path.join('/home/cf/Dropbox/thesis/figures_methods', 'comp_optimization_algorithms.png'))
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

    def plot_fitness_by_generation(self, save_dir, n_params, trial, save_dir_plots):

        pl.figure()

        # change color cycle
        cmap = pl.get_cmap('jet')
        colors = cmap(np.linspace(0.1, 0.9, len(self.methods)))
        for m, method in enumerate(methods):

            # read individuals_file
            save_dir_trial = save_dir[n_params-1] + method + '/' + 'trial' + str(trial) + '/'
            path = save_dir_trial + 'individuals_file.csv'
            individuals_file = pd.read_csv(path,
                                           dtype={'generation': np.int64, 'number': np.int64, 'fitness': np.float64,
                                                  'candidate': str})
            # find best candidate
            n_generations = individuals_file.generation.iloc[-1]
            if not os.path.exists(save_dir_plots):
                os.makedirs(save_dir_plots)

            # best fitness
            fitness_min = np.zeros(n_generations)
            for generation in range(n_generations):
                fitness_min[generation] = np.min(
                    individuals_file.fitness[individuals_file.generation.isin([generation])])

            # plot
            pl.plot(range(n_generations), fitness_min, color=colors[m], label=method)
        pl.xlabel('Generation', fontsize=16)
        pl.ylabel('Fitness', fontsize=16)
        pl.legend()
        pl.savefig(save_dir_plots+'nparams'+str(n_params)+'_trial'+str(trial)+'.png')
        pl.show()

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

    def plot_candidate(self, n_param, trial, method):
        with open(self.save_dirs[n_param-1]+'specification/trial'+str(trial)+'/problem.json', 'r') as f:
            problem = json.load(f)
        data = pd.read_csv(problem['data_dir'])
        simulation_params = extract_simulation_params(data, **{'celsius': problem['simulation_params']['celsius']})

        # create cell
        candidate, _ = self.get_candidate_and_fitness(self.save_dirs[n_param-1]+method+'/', trial)
        model_dir = self.save_dirs[n_param-1]+'specification/trial'+str(trial)+'/cell.json'
        _, _, variable_keys = get_lowerbound_upperbound_keys(problem['variables'])
        cell = Cell.from_modeldir(model_dir, mechanism_dir=problem['mechanism_dir'])
        cell.insert_mechanisms(variable_keys)
        for i in range(len(candidate)):
            for path in variable_keys[i]:
                cell.update_attr(path, candidate[i])

        v_candidate, t_candidate = iclamp(cell, **simulation_params)

        pl.figure()
        pl.plot(data.t, data.v, 'k', label='data')
        pl.plot(t_candidate, v_candidate, 'r', label='fit')
        pl.legend(fontsize=14)
        pl.show()

        return v_candidate, t_candidate


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
    #evaluator.save_statistics()
    #evaluator.plot_statistic('rms(param)', 'mean')
    #evaluator.plot_statistic('rms(param)', 'min')
    #evaluator.plot_statistic('rms(v)', 'mean')
    #evaluator.plot_statistic('rms(v)', 'min')
    #evaluator.hist_mean_error_variable_comb(save_dirs[0], 1)
    #evaluator.plot_2d_mean_error_variable_comb()
    evaluator.plot_statistic_without_table()

    #save_dir_fitness_by_generation = save_dir_statistics + 'fitness_by_generation/'
    #p = 10
    #trial = 4
    #evaluator.plot_fitness_by_generation(save_dirs, p, trial, save_dir_fitness_by_generation)

    #evaluator.plot_candidate(10, 4, 'L-BFGS-B')

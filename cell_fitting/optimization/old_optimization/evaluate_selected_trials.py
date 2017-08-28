import json


def select_trials(save_dir, variables_not_wanted):
    selected_trials = list()
    for trial in range(n_trials):
        with open(save_dir+'specification/trial'+str(trial)+'/problem.json', 'r') as f:
            problem_params = json.load(f)
        variable_names = [problem_params['variables'][i][2][0][-1] for i in range(len(problem_params['variables']))]
        if not any([variable_not_wanted in variable_names for variable_not_wanted in variables_not_wanted]):
            selected_trials.append(trial)
    return selected_trials


if __name__ == '__main__':
    from optimization.old_optimization.evaluation import Evaluator

    # parameter
    methods = ['DEA', 'SA', 'PSO', 'L-BFGS-B', 'Nelder-Mead', 'random']
    method_types = ['ec', 'ec', 'swarm', 'gradient_based', 'simplex', 'random']
    save_dir_statistics = '../../results/algorithms_on_hhcell/statistic_without_beta_m_f/'
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

    variables_not_wanted = ['beta_n_k']

    selected_trials_per_savedir = list()
    for save_dir in save_dirs:
        selected_trials_per_savedir.append(select_trials(save_dir, variables_not_wanted))

    for s, save_dir in enumerate(save_dirs):
        print 100 - len(selected_trials_per_savedir[s])

    #evaluator = Evaluator(save_dir_statistics, save_dirs, n_trials, methods)
    #evaluator.save_error_weights_and_best_fitness(selected_trials=selected_trials_per_savedir,
    #                                              name_error_weights='error_weights_without_beta_m_f',
    #                                              name_best_fitness='best_fitness_without_beta_m_f')
    #evaluator.save_statistics(name_error_weights='error_weights_without_beta_m_f',
    #                          name_best_fitness='best_fitness_without_beta_m_f')
    #evaluator.plot_statistic('rms(param)', 'mean')
    #evaluator.plot_statistic('rms(param)', 'min')
    #evaluator.plot_statistic('rms(v)', 'mean')
    #evaluator.plot_statistic('rms(v)', 'min')
    #evaluator.hist_mean_error_variable_comb(save_dirs[0], 1, name_error_weights='error_weights_without_beta_m_f')
    #evaluator.plot_2d_mean_error_variable_comb(name_error_weights='error_weights_without_beta_m_f')
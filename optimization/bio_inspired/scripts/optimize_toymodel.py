from optimization.bio_inspired.optimize import optimize

__author__ = 'caro'


# parameter
save_dir = '../../../results/bio_inspired/test_algorithms/increase_params/2param/'
n_trials = 20

pop_size = 1000
max_generations = 100
idx = 0
methods = ['DEA', 'SA', 'GA', 'EDA', 'PSO']
method_types = ['ec', 'ec', 'ec', 'ec', 'swarm']
method_args = [
    {'pop_size': pop_size, 'num_selected': 670, 'tournament_size': 360, 'crossover_rate': 0.57,
     'mutation_rate': 0.52, 'gaussian_mean': 0.0, 'gaussian_stdev': 0.21, 'max_generations': max_generations},
    {'temperature': 524.25, 'cooling_rate': 0.51, 'mutation_rate': 0.68, 'gaussian_mean': 0.0, 'gaussian_stdev': 0.20,
     'max_generations': max_generations * pop_size},
    {'pop_size': pop_size, 'num_selected': pop_size, 'crossover_rate': 0.44, 'num_crossover_points': 2,
     'mutation_rate': 0.53, 'gaussian_mean': 0.0, 'gaussian_stdev': 0.10, 'num_elites': 280,
     'max_generations': max_generations},
    {'pop_size': pop_size, 'num_selected': 490, 'num_offspring': pop_size, 'num_elites': 270,
     'max_generations': max_generations},
    {'pop_size': pop_size, 'inertia': 0.43, 'cognitive_rate': 1.44, 'social_rate': 1.57,
     'max_generations': max_generations}
    ]

method = methods[idx]
method_type = method_types[idx]
method_args = method_args[idx]

variables = [
            [0, 2.5, [['soma', '0.5', 'na8st', 'gbar']]],
            [0, 2.5, [['soma', '0.5', 'kdr', 'gbar']]]
            #[0, 2.5, [['soma', '0.5', 'na8st', 'gbar']]]
            ]

params = {
          'name': 'CellFitProblem',
          'maximize': False,
          'normalize': True,
          'model_dir': '../../../model/cells/toymodel3.json',
          'mechanism_dir': '../../../model/channels/schmidthieber',
          'variables': variables,
          'data_dir': '../../../data/toymodels/toymodel3/ramp.csv',
          'get_var_to_fit': 'get_v',
          'fitnessweights': [1.0],
          'errfun': 'rms',
          'insert_mechanisms': True
         }

optimize(save_dir, n_trials, params, method, method_type, method_args, params['name'])
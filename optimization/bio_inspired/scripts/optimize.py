from optimization.bio_inspired.parametersearch import optimize

__author__ = 'caro'


# parameter
save_dir = '../../../results/bio_inspired/DAP_fromAP/'
n_trials = 1

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
     'max_generation': max_generations},
    {'pop_size': pop_size, 'num_selected': 490, 'num_offspring': pop_size, 'num_elites': 270,
     'max_generations': max_generations},
    {'pop_size': pop_size, 'inertia': 0.43, 'cognitive_rate': 1.44, 'social_rate': 1.57,
     'max_generations': max_generations}
    ]

method = methods[idx]
method_type = method_types[idx]
method_args = method_args[idx]

variables = [
            [0, 1.5, [['soma', '0.5', 'na8st', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'nap', 'gbar']]],
            [0, 1.0, [['soma', '0.5', 'kdr', 'gbar']]],
            [0, 1.0, [['soma', '0.5', 'kap', 'gbar']]],
            [0, 0.1, [['soma', '0.5', 'ih', 'gslowbar']]],
            [0, 0.1, [['soma', '0.5', 'ih', 'gfastbar']]],
            [0, 0.1, [['soma', '0.5', 'km', 'gbar']]],
            [0, 0.5, [['soma', '0.5', 'pas', 'g']]]
            ]

params = {
          'maximize': False,
          'normalize': True,
          'model_dir': '../../../model/cells/pointmodel.json',
          'mechanism_dir': '../../../model/channels/schmidthieber',
          'variables': variables,
          'data_dir': '../../../data/2015_08_11d/ramp/dap.csv',
          'get_var_to_fit': 'get_v',
          'fitnessweights': [1.0],
          'errfun': 'rms',
          'insert_mechanisms': True
         }

optimize(save_dir, n_trials, params, method, method_type, method_args)
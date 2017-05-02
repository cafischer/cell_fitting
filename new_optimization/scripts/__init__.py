from new_optimization import OptimizationSettings, AlgorithmSettings
from new_optimization.optimizer import OptimizerFactory
import os
from time import time
import json
from memory_profiler import memory_usage
from util import merge_dicts
from optimization.bio_inspired.generators import get_random_numbers_in_bounds
from new_optimization import create_pseudo_random_number_generator
from nrn_wrapper import load_mechanism_dir


def optimize(optimization_settings_dict, algorithm_settings_dict, save_memory_usage=False):
    algorithm_settings_dict['save_dir'] = os.path.join(algorithm_settings_dict['save_dir'],
                                                       algorithm_settings_dict['algorithm_name'])
    optimization_settings = OptimizationSettings(**optimization_settings_dict)
    algorithm_settings = AlgorithmSettings(**algorithm_settings_dict)

    optimizer = OptimizerFactory().make_optimizer(optimization_settings, algorithm_settings)
    optimizer.save(algorithm_settings_dict['save_dir'])

    start_time = time()
    if save_memory_usage:
        mem_usage = memory_usage(optimizer.optimize())
        with open(os.path.join(algorithm_settings_dict['save_dir'], 'memory_usage.txt'), 'w') as f:
            f.write(str(max(mem_usage)))
    else:
        optimizer.optimize()
    end_time = time()
    duration = end_time - start_time
    with open(os.path.join(algorithm_settings_dict['save_dir'], 'duration.txt'), 'w') as f:
        f.write(str(duration))


def optimize_hyperparameters(hyperparameter_dict, optimization_settings_dict, algorithm_settings_dict):

    mechanism_dir = optimization_settings_dict['fitter_params']['mechanism_dir']
    optimization_settings_dict['fitter_params']['mechanism_dir'] = None
    load_mechanism_dir(mechanism_dir)

    random_generator = create_pseudo_random_number_generator(hyperparameter_dict['seed'])
    save_dir = algorithm_settings_dict['save_dir']
    if not os.path.exists(os.path.join(save_dir, algorithm_settings_dict['algorithm_name'])):
        os.makedirs(os.path.join(save_dir, algorithm_settings_dict['algorithm_name']))
    with open(os.path.join(os.path.join(save_dir, algorithm_settings_dict['algorithm_name']),
                           'hyperparameter_settings.json'), 'w') as f:
        json.dump(hyperparameter_dict, f)

    for i in range(hyperparameter_dict['n_samples']):
        hyperparams = dict(zip(hyperparameter_dict['parameter_names'], get_random_numbers_in_bounds(
                                                         random_generator,
                                                         hyperparameter_dict['lower_bounds'],
                                                         hyperparameter_dict['upper_bounds'], None)))

        algorithm_settings_dict['algorithm_params'] = merge_dicts(algorithm_settings_dict['algorithm_params'],
                                                                     hyperparams)
        algorithm_settings_dict['save_dir'] = os.path.join(save_dir) #, str(i))
        optimize(optimization_settings_dict, algorithm_settings_dict)
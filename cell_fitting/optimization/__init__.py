import json
import os
from random import Random

from cell_fitting.optimization.bio_inspired import generators
from cell_fitting.optimization.fitter import FitterFactory


class OptimizationSettings:

    def __init__(self, maximize, n_candidates, stop_criterion, seed, generator, bounds, fitter_params, extra_args):
        self.maximize = maximize
        self.n_candidates = n_candidates
        self.stop_criterion = stop_criterion
        self.seed = seed
        self.generator = generator
        self.bounds = bounds
        self.fitter = FitterFactory().make_fitter(fitter_params)
        self.extra_args = extra_args

    @classmethod
    def load(cls, load_file):
        settings = json.load(load_file)
        return cls(**settings)

    def to_dict(self):
        return {'maximize': self.maximize, 'n_candidates': self.n_candidates, 'stop_criterion': self.stop_criterion,
                'seed': self.seed, 'generator': self.generator, 'bounds': self.bounds,
                'fitter_params': self.fitter.to_dict(), 'extra_args': self.extra_args}

    def save(self, save_file):
        json.dump(self.to_dict(), save_file, indent=4)


class AlgorithmSettings:

    def __init__(self, algorithm_name, algorithm_params, optimization_params, normalize, save_dir):
        self.algorithm_name = algorithm_name
        self.algorithm_params = algorithm_params
        self.optimization_params = optimization_params
        self.normalize = normalize
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    @classmethod
    def load(cls, load_file):
        settings = json.load(load_file)
        return cls(**settings)

    def to_dict(self):
        return {'algorithm_name': self.algorithm_name, 'algorithm_params': self.algorithm_params,
                'optimization_params': self.optimization_params, 'normalize': self.normalize,
                'save_dir': self.save_dir}

    def save(self, save_file):
        json.dump(self.to_dict(), save_file, indent=4)


def create_pseudo_random_number_generator(seed):
    pseudo_random_number_generator = Random()
    pseudo_random_number_generator.seed(seed)
    return pseudo_random_number_generator


def generate_candidates(generator_name, lower_bounds, upper_bounds, seed, n_candidates):
    generator = getattr(generators, generator_name)
    random = create_pseudo_random_number_generator(seed)
    initial_candidates = [generator(random, lower_bounds, upper_bounds, None)
                          for i in range(n_candidates)]
    return initial_candidates
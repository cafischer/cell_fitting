import abc
import json
from cell_fitting.optimization import generate_initial_candidates


class Optimizer:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, optimization_settings, algorithm_settings):
        self.optimization_settings = optimization_settings
        self.algorithm_settings = algorithm_settings

    @abc.abstractmethod
    def optimize(self):
        pass

    def save(self, save_dir):
        with open(save_dir + '/optimization_settings.json', 'w') as f:
            self.optimization_settings.save(f)
        with open(save_dir + '/algorithm_settings.json', 'w') as f:
            self.algorithm_settings.save(f)

        if getattr(self.optimization_settings.fitter, 'model_dir', None) is not None:
            with open(self.optimization_settings.fitter.model_dir, 'r') as f1:
                cell = json.load(f1)
                with open(save_dir + '/cell.json', 'w') as f2:
                    json.dump(cell, f2, indent=4)

    def generate_initial_candidates(self):
        return self.optimization_settings.extra_args.pop('init_candidates',
                                             generate_initial_candidates(
                                                                    self.optimization_settings.generator,
                                                                    self.optimization_settings.bounds['lower_bounds'],
                                                                    self.optimization_settings.bounds['upper_bounds'],
                                                                    self.optimization_settings.seed,
                                                                    self.optimization_settings.n_candidates))
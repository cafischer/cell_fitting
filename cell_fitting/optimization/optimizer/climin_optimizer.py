import functools
import numdifftools as nd
import numpy as np
from cell_fitting.optimization.optimizer.optimizer_interface import Optimizer
from cell_fitting.optimization import generate_initial_candidates
import climin
import copy
import os


class CliminOptimizer(Optimizer):
    algorithm_name_dict = {'rmsprop': 'RmsProp', 'adam': 'Adam', 'adadelta': 'Adadelta'}

    def __init__(self, optimization_settings, algorithm_settings):
        super(CliminOptimizer, self).__init__(optimization_settings, algorithm_settings)
        self.individuals_file = open(os.path.join(self.algorithm_settings.save_dir, 'candidates.csv'), 'w')
        self.individuals_file.write('{0},{1},{2},{3}\n'.format('generation', 'id', 'fitness', 'candidate'))
        self.optimization_algorithm = getattr(getattr(climin, algorithm_settings.algorithm_name),
                                              CliminOptimizer.algorithm_name_dict[algorithm_settings.algorithm_name])

        self.algorithm_params = self.algorithm_settings.algorithm_params
        self.initial_candidates = self.generate_initial_candidates()
        self.bounds = self.transform_bounds(self.optimization_settings.bounds)
        self.fun = functools.partial(self.optimization_settings.fitter.evaluate_fitness, args=None)
        self.step = self.algorithm_settings.algorithm_params.get('step', 1e-8)

        def jac(candidate):
            jac_value = np.squeeze(nd.Jacobian(self.fun, step=self.step, method='central')(candidate))
            jac_value[np.isnan(jac_value)] = 0
            return jac_value
        self.jac = jac

    @staticmethod
    def transform_bounds(bounds):
        bounds_transformed = list()
        for lb, ub in zip(bounds['lower_bounds'], bounds['upper_bounds']):
            bounds_transformed.append((lb, ub))
        return bounds_transformed

    def optimize(self):
        for id, candidate in enumerate(self.initial_candidates):
            candidates = list()
            optimization_algorithm = self.optimization_algorithm(np.array(candidate), self.jac, **self.algorithm_params)
            for info in optimization_algorithm:
                candidates.append(copy.copy(optimization_algorithm.wrt))
                if info['n_iter'] >= self.optimization_settings.stop_criterion[1]:
                    break
            self.save_candidates(candidates, id)

    def save_candidates(self, candidates, id):
        fitness = [self.fun(c) for c in candidates]
        candidates = [str(c).replace('[', '').replace(']', '').replace(',', '') for c in candidates]

        self.write_file(candidates, id, fitness)

    def write_file(self, candidates, id, fitness):
        for i in range(len(candidates)):
            self.individuals_file.write('{0},{1},{2},{3}\n'.format(i, id, fitness[i], candidates[i]))
        self.individuals_file.flush()
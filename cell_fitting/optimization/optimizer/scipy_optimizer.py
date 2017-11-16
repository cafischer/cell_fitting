import functools
import os

import numdifftools as nd
import pandas as pd
from scipy.optimize import minimize

from cell_fitting.optimization import generate_initial_candidates
from cell_fitting.optimization.optimizer.optimizer_interface import Optimizer
from cell_fitting.util import merge_dicts


class ScipyOptimizer(Optimizer):

    def __init__(self, optimization_settings, algorithm_settings):
        super(ScipyOptimizer, self).__init__(optimization_settings, algorithm_settings)

        self.individuals_file = open(os.path.join(self.algorithm_settings.save_dir, 'candidates.csv'), 'w')
        self.individuals_file.write('{0},{1},{2},{3},{4},{5}\n'.format('generation', 'id', 'fitness', 'candidate',
                                                                       'success', 'termination'))

        self.args = self.set_args()
        self.initial_candidates = self.generate_initial_candidates()
        self.bounds = self.transform_bounds(self.optimization_settings.bounds)
        self.fun = functools.partial(self.optimization_settings.fitter.evaluate_fitness, args=None)
        self.step = self.algorithm_settings.algorithm_params.get('step', 1e-8)

        def jac(candidate):
            jac_value = np.squeeze(nd.Jacobian(self.fun, step=self.step, method='central')(candidate))
            jac_value[np.isnan(jac_value)] = 0
            return jac_value

        def hess(candidate):
            hess_value = nd.Hessian(self.fun, step=self.step, method='central')(candidate)
            hess_value[np.isnan(hess_value)] = 0
            return hess_value

        self.jac = jac
        self.hess = hess

    def set_args(self):
        if self.algorithm_settings.optimization_params is None:
            self.algorithm_settings.optimization_params = {}
        args = merge_dicts(self.set_stop_criterion(), self.algorithm_settings.optimization_params)
        args['options'] = merge_dicts(args['options'], self.algorithm_settings.algorithm_params)
        return args

    def set_stop_criterion(self):
        args = dict()
        algorithm_name = self.algorithm_settings.algorithm_name
        if self.optimization_settings.stop_criterion[0] == 'generation_termination':
            if algorithm_name == 'Nelder-Mead':
                options = {'maxiter': self.optimization_settings.stop_criterion[1] + 1}
            elif algorithm_name == 'L-BFGS-B':
                options = {'maxiter': self.optimization_settings.stop_criterion[1] - 1}
            else:
                options = {'maxiter': self.optimization_settings.stop_criterion[1]}
            args['options'] = options
        elif self.optimization_settings.stop_criterion[0] == 'evaluation_termination':
            options = {'maxfun': self.optimization_settings.stop_criterion[1]}  # TODO depends on method
            args['options'] = options
        elif self.optimization_settings.stop_criterion[0] == 'tolerance_termination':
            args['tol'] = self.optimization_settings.stop_criterion[1]
        else:
            raise ValueError('Unknown stop criterion!')
        return args

    @staticmethod
    def transform_bounds(bounds):
        bounds_transformed = list()
        for lb, ub in zip(bounds['lower_bounds'], bounds['upper_bounds']):
            bounds_transformed.append((lb, ub))
        return bounds_transformed

    def optimize(self):
        for id, candidate in enumerate(self.initial_candidates):
            candidates = list()
            callback = functools.partial(self.store_candidates, candidates=candidates)
            callback(candidate)

            result = minimize(fun=self.fun, x0=np.array(candidate), method=self.algorithm_settings.algorithm_name, jac=self.jac,
                     hess=self.hess, bounds=self.bounds, callback=callback, **self.args)
            self.save_candidates(candidates, id, result.success, result.message)

    def store_candidates(self, candidate, candidates):
        candidates.append(list(candidate))

    def save_candidates(self, candidates, id, success_end, termination_end):
        success = ['']*len(candidates)
        success[-1] = success_end
        termination = ['']*len(candidates)
        termination[-1] = termination_end.replace(',', '')
        fitness = [self.fun(c) for c in candidates]
        candidates = [str(l).replace('[', '').replace(']', '').replace(',', '') for l in candidates]

        self.write_file(candidates, id, fitness, success, termination)

    def write_file(self, candidates, id, fitness, success, termination):
        for i in range(len(candidates)):
            self.individuals_file.write('{0},{1},{2},{3},{4},{5}\n'.format(i, id, fitness[i], candidates[i],
                                                                           success[i], termination[i]))
        self.individuals_file.flush()


class ScipyOptimizerInitBounds(ScipyOptimizer):

    def __init__(self, optimization_settings, algorithm_settings):
        self.init_bounds = optimization_settings.extra_args.pop('init_bounds')
        super(ScipyOptimizerInitBounds, self).__init__(optimization_settings, algorithm_settings)

    def generate_initial_candidates(self):
        return generate_initial_candidates(
                                            self.optimization_settings.generator,
                                            self.init_bounds['lower_bounds'],
                                            self.init_bounds['upper_bounds'],
                                            self.optimization_settings.seed,
                                            self.optimization_settings.n_candidates)


from cell_fitting.optimization.gradient_based import *


class ScipyCGOptimizer(ScipyOptimizer):

    def __init__(self, optimization_settings, algorithm_settings):
        super(ScipyCGOptimizer, self).__init__(optimization_settings, algorithm_settings)

    def optimize(self):

        for id, candidate in enumerate(self.initial_candidates):
            callback = functools.partial(self.store_candidates, id=id)
            self.num_generations = 0

            xs = conjugate_gradient(self.fun, self.jac, candidate, self.args['options']['maxiter'], c1=1e-4, c2=0.4)
            for x in xs:
                self.store_candidates(x, id)
            self.save_candidates()

    def store_candidates(self, candidate, id):
        fitness = self.fun(candidate)
        self.candidates.append([self.num_generations, id, fitness,
                                str(list(candidate)).replace(',', '').replace('[', '').replace(']', '')])
        self.num_generations += 1

    def save_candidates(self):
        individuals_data = pd.DataFrame(self.candidates, columns=['generation', 'id', 'fitness', 'candidate'])
        individuals_data = individuals_data.groupby('generation').apply(lambda x: x.sort_values(['fitness']))
        with open(self.algorithm_settings.save_dir + 'candidates.csv', 'w') as f:
            individuals_data.to_csv(f, header=True, index=False)
import functools
import numdifftools as nd
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from new_optimization import *
from new_optimization.optimizer.optimizer_interface import Optimizer
from optimization.bio_inspired import generators
from util import merge_dicts


class ScipyOptimizer(Optimizer):

    def __init__(self, optimization_settings, algorithm_settings):
        super(ScipyOptimizer, self).__init__(optimization_settings, algorithm_settings)

        self.args = self.set_args()
        self.initial_candidates = self.generate_initial_candidates(self.optimization_settings.generator,
                                                                   self.optimization_settings.seed)
        self.bounds = self.transform_bounds(self.optimization_settings.bounds)
        self.fun = functools.partial(self.optimization_settings.fitter.evaluate_fitness, args=None)
        self.step = self.algorithm_settings.algorithm_params.get('step', 1e-8)

        def jac(candidate):
            jac_value = nd.Jacobian(self.fun, step=self.step, method='central')(candidate)[0]
            jac_value[np.isnan(jac_value)] = 0
            return jac_value

        def hess(candidate):
            hess_value = nd.Hessian(self.fun, step=self.step, method='central')(candidate)
            hess_value[np.isnan(hess_value)] = 0
            return hess_value

        self.jac = jac
        self.hess = hess

        self.candidates = list()
        self.num_generations = 0

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

    def generate_initial_candidates(self, generator_name, seed):
        generator = getattr(generators, generator_name)
        random = create_pseudo_random_number_generator(seed)
        initial_candidates = [generator(random, self.optimization_settings.bounds['lower_bounds'],
                                        self.optimization_settings.bounds['upper_bounds'], None)
                              for i in range(self.optimization_settings.n_candidates)]
        return initial_candidates

    def transform_bounds(self, bounds):
        bounds_transformed = list()
        for lb, ub in zip(bounds['lower_bounds'], bounds['upper_bounds']):
            bounds_transformed.append((lb, ub))
        return bounds_transformed

    def optimize(self):

        for id, candidate in enumerate(self.initial_candidates):
            callback = functools.partial(self.store_candidates, id=id)
            self.num_generations = 0
            self.store_candidates(candidate, id)

            result = minimize(fun=self.fun, x0=candidate, method=self.algorithm_settings.algorithm_name, jac=self.jac,
                     hess=self.hess, bounds=self.bounds, callback=callback, **self.args)
            self.candidates[-1][4] = result.success
            self.candidates[-1][5] = result.message
            self.save_candidates()

        # scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None,
        # constraints=(), tol=None, callback=None, options=None)[source]

    def store_candidates(self, candidate, id):
        fitness = self.fun(candidate)
        #self.candidates.append([self.num_generations, id, fitness,
        #                        str(list(candidate)).replace(',', '').replace('[', '').replace(']', '')])
        self.candidates.append([self.num_generations, id, fitness,
                                str(list(candidate)).replace(',', '').replace('[', '').replace(']', ''), '', ''])
        self.num_generations += 1

    def save_candidates(self):
        individuals_data = pd.DataFrame(self.candidates, columns=['generation', 'id', 'fitness', 'candidate', 'success',
                                                                  'termination'])
        individuals_data = individuals_data.groupby('generation').apply(lambda x: x.sort_values(['fitness']))
        with open(self.algorithm_settings.save_dir + 'candidates.csv', 'w') as f:
            individuals_data.to_csv(f, header=True, index=False)


class ScipyOptimizerInitBounds(ScipyOptimizer):

    def __init__(self, optimization_settings, algorithm_settings):
        self.init_bounds = algorithm_settings.optimization_params['init_bounds']
        del algorithm_settings.optimization_params['init_bounds']
        super(ScipyOptimizerInitBounds, self).__init__(optimization_settings, algorithm_settings)

    def generate_initial_candidates(self, generator_name, seed):
        generator = getattr(generators, generator_name)
        random = create_pseudo_random_number_generator(seed)
        initial_candidates = [generator(random, self.init_bounds['lower_bounds'],
                                        self.init_bounds['upper_bounds'], None)
                              for i in range(self.optimization_settings.n_candidates)]
        return initial_candidates


from optimization.gradient_based import *


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
from new_optimization import *
from scipy.optimize import minimize
from optimization.gradient_based import numerical_gradient
from optimization.bio_inspired.inspyred_extension import generators
from util import merge_dicts
import functools
import pandas as pd


class ScipyOptimizer(Optimizer):

    def __init__(self, optimization_settings, algorithm_settings):
        super(ScipyOptimizer, self).__init__(optimization_settings, algorithm_settings)

        self.args = self.set_args()
        self.initial_candidates = self.generate_initial_candidates(self.optimization_settings.generator,
                                                                   self.optimization_settings.seed)
        self.bounds = self.transform_bounds(self.optimization_settings.bounds)

        self.fun = functools.partial(self.optimization_settings.fitter.evaluate_fitness, args=None)
        self.funprime = functools.partial(numerical_gradient, f=self.fun, method='central')

        self.candidates = list()
        self.num_generations = 0

    def set_args(self):
        if self.algorithm_settings.optimization_params is None:
            self.algorithm_settings.optimization_params = {}
        args = merge_dicts(self.set_stop_criterion(), self.algorithm_settings.optimization_params)
        return args

    def set_stop_criterion(self):
        args = dict()
        if self.optimization_settings.stop_criterion[0] == 'generation_termination':
            if self.algorithm_settings.algorithm_name == 'Nelder-Mead':
                options = {'maxiter': self.optimization_settings.stop_criterion[1] + 1}
            elif self.algorithm_settings.algorithm_name == 'L-BFGS-B':
                options = {'maxiter': self.optimization_settings.stop_criterion[1]-1}
            else:
                raise ValueError('maxiter not implemented for this method!')
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

            minimize(fun=self.fun, x0=candidate, method=self.algorithm_settings.algorithm_name, jac=self.funprime,
                     bounds=self.bounds, callback=callback, **self.args)

            self.save_candidates()

        # scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None,
        # constraints=(), tol=None, callback=None, options=None)[source]

    def store_candidates(self, candidate, id):
        fitness = self.fun(candidate)
        self.candidates.append([self.num_generations, id, fitness,
                                str(candidate).replace(',', '').replace('[', '').replace(']', '')])
        self.num_generations += 1

    def save_candidates(self):
        individuals_data = pd.DataFrame(self.candidates, columns=['generation', 'id', 'fitness', 'candidate'])
        individuals_data = individuals_data.groupby('generation').apply(lambda x: x.sort_values(['fitness']))
        with open(self.algorithm_settings.save_dir + 'candidates.csv', 'w') as f:
            individuals_data.to_csv(f, header=True, index=False)
import functools
import numdifftools as nd
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from new_optimization import create_pseudo_random_number_generator
from new_optimization.optimizer.optimizer_interface import Optimizer
from optimization.bio_inspired import generators
from util import merge_dicts
import multiprocessing


class ScipyOptimizer(Optimizer):

    def __init__(self, optimization_settings, algorithm_settings):
        super(ScipyOptimizer, self).__init__(optimization_settings, algorithm_settings)

        self.individuals_file = open(self.algorithm_settings.save_dir + 'candidates.csv', 'w')
        self.individuals_file.write('{0},{1},{2},{3},{4},{5}\n'.format('generation', 'id', 'fitness', 'candidate',
                                                                       'success', 'termination'))
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

    def optimize(self):  # with multiprocessing
        queue = multiprocessing.Queue()
        work_queue = multiprocessing.Queue()
        processes = list()

        # put all jobs in the queue
        for id, candidate in enumerate(self.initial_candidates):
            work_queue.put((id, candidate))

        # start processes
        for core in range(multiprocessing.cpu_count()):
            p = multiprocessing.Process(target=self.work, args=(work_queue, queue))
            p.start()
            processes.append(p)

        # save results
        counter = 0
        while True:
            candidates, id, success, message = queue.get()
            counter += 1
            self.save_candidates(candidates, id, success, message)
            if counter == len(self.initial_candidates):
                break
        # stop
        for p in processes:
            p.join()

    def work(self, working_queue, queue):
        while not working_queue.empty():
            id, candidate = working_queue.get()
            self.optimize_single_candidate(id, candidate, queue)

    def optimize_single_candidate(self, id, candidate, queue):
        candidates = list()
        callback = functools.partial(self.store_candidates, candidates=candidates)
        callback(candidate)  # store first candidate

        result = minimize(fun=self.fun, x0=candidate, method=self.algorithm_settings.algorithm_name, jac=self.jac,
                          hess=self.hess, bounds=self.bounds, callback=callback, **self.args)
        queue.put((candidates, id, result.success, result.message))
        return candidates

    def store_candidates(self, candidate, candidates):
        candidates.append(list(candidate))

    def save_candidates(self, candidates, id, success_end, termination_end):
        success = ['']*len(candidates)
        success[-1] = success_end
        termination = ['']*len(candidates)
        termination[-1] = termination_end
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
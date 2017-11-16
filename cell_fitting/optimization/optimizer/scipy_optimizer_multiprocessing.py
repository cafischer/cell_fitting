import functools
import multiprocessing

import numpy as np
from scipy.optimize import minimize

from cell_fitting.optimization import create_pseudo_random_number_generator
from cell_fitting.optimization.bio_inspired import generators
from cell_fitting.optimization.optimizer.scipy_optimizer import ScipyOptimizer


class ScipyOptimizerMultiprocessing(ScipyOptimizer):

    def __init__(self, optimization_settings, algorithm_settings):
        super(ScipyOptimizerMultiprocessing, self).__init__(optimization_settings, algorithm_settings)
        self.n_cores = optimization_settings.extra_args.pop('n_cores', multiprocessing.cpu_count())

    def optimize(self):  # with multiprocessing
        queue = multiprocessing.Queue()
        work_queue = multiprocessing.Queue()
        processes = list()

        # put all jobs in the queue
        for id, candidate in enumerate(self.initial_candidates):
            work_queue.put((id, candidate))

        # start processes
        for core in range(self.n_cores):
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

        result = minimize(fun=self.fun, x0=np.array(candidate), method=self.algorithm_settings.algorithm_name, jac=self.jac,
                          hess=self.hess, bounds=self.bounds, callback=callback, **self.args)
        queue.put((candidates, id, result.success, result.message))
        return candidates


class ScipyOptimizerInitBoundsMultiprocessing(ScipyOptimizerMultiprocessing):

    def __init__(self, optimization_settings, algorithm_settings):
        self.init_bounds = optimization_settings.extra_args.pop('init_bounds')
        super(ScipyOptimizerInitBoundsMultiprocessing, self).__init__(optimization_settings, algorithm_settings)

    def generate_initial_candidates(self, generator_name, seed):
        generator = getattr(generators, generator_name)
        random = create_pseudo_random_number_generator(seed)
        initial_candidates = [generator(random, self.init_bounds['lower_bounds'],
                                        self.init_bounds['upper_bounds'], None)
                              for i in range(self.optimization_settings.n_candidates)]
        return initial_candidates
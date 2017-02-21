import functools
import numdifftools as nd
import numpy as np
from new_optimization import create_pseudo_random_number_generator
from new_optimization.optimizer.optimizer_interface import Optimizer
from optimization.bio_inspired import generators
import multiprocessing
import climin
import copy


class CliminOptimizer(Optimizer):
    algorithm_name_dict = {'rmsprop': 'RmsProp', 'adam': 'Adam', 'adadelta': 'Adadelta'}

    def __init__(self, optimization_settings, algorithm_settings):
        super(CliminOptimizer, self).__init__(optimization_settings, algorithm_settings)
        self.individuals_file = open(self.algorithm_settings.save_dir + 'candidates.csv', 'w')
        self.individuals_file.write('{0},{1},{2},{3}\n'.format('generation', 'id', 'fitness', 'candidate'))
        self.optimization_algorithm = getattr(getattr(climin, algorithm_settings.algorithm_name),
                                              CliminOptimizer.algorithm_name_dict[algorithm_settings.algorithm_name])

        self.algorithm_params = self.algorithm_settings.algorithm_params
        self.initial_candidates = self.generate_initial_candidates(self.optimization_settings.generator,
                                                                   self.optimization_settings.seed)
        self.bounds = self.transform_bounds(self.optimization_settings.bounds)
        self.fun = functools.partial(self.optimization_settings.fitter.evaluate_fitness, args=None)
        self.step = self.algorithm_settings.algorithm_params.get('step', 1e-8)

        def jac(candidate):
            jac_value = nd.Jacobian(self.fun, step=self.step, method='central')(candidate)[0]
            jac_value[np.isnan(jac_value)] = 0
            return jac_value
        self.jac = jac

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
            work_queue.put((id, np.array(candidate)))

        # start processes
        for core in range(multiprocessing.cpu_count()):
            p = multiprocessing.Process(target=self.work, args=(work_queue, queue))
            p.start()
            processes.append(p)

        # save results
        counter = 0
        while True:
            candidates, id = queue.get()
            counter += 1
            self.save_candidates(candidates, id)
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
        candidates.append(copy.copy(candidate))
        optimization_algorithm = self.optimization_algorithm(candidate, self.jac, **self.algorithm_params)
        for info in optimization_algorithm:
            candidates.append(copy.copy(optimization_algorithm.wrt))
            if info['n_iter'] >= self.optimization_settings.stop_criterion[1]:
                break
        print candidates
        queue.put((candidates, id))
        return candidates

    def save_candidates(self, candidates, id):
        fitness = [self.fun(c) for c in candidates]
        candidates = [str(list(c)).replace('[', '').replace(']', '').replace(',', '') for c in candidates]
        self.write_file(candidates, id, fitness)

    def write_file(self, candidates, id, fitness):
        for i in range(len(candidates)):
            self.individuals_file.write('{0},{1},{2},{3}\n'.format(i, id, fitness[i], candidates[i]))
        self.individuals_file.flush()
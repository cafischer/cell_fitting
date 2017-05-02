import numpy as np
from new_optimization.optimizer.climin_optimizer import CliminOptimizer
import multiprocessing
import copy


class CliminOptimizerMultiprocessing(CliminOptimizer):
    algorithm_name_dict = {'rmsprop': 'RmsProp', 'adam': 'Adam', 'adadelta': 'Adadelta'}

    def __init__(self, optimization_settings, algorithm_settings):
        super(CliminOptimizerMultiprocessing, self).__init__(optimization_settings, algorithm_settings)
        self.n_cores = optimization_settings.extra_args.pop('n_cores', multiprocessing.cpu_count())

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
        optimization_algorithm = self.optimization_algorithm(np.array(candidate), self.jac, **self.algorithm_params)
        for info in optimization_algorithm:
            candidates.append(copy.copy(optimization_algorithm.wrt))
            if info['n_iter'] >= self.optimization_settings.stop_criterion[1]:
                break
        queue.put((candidates, id))
        return candidates
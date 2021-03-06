import functools
import os

import inspyred
import pandas as pd
from cell_fitting.optimization.bio_inspired import generators, observers

from cell_fitting.optimization import create_pseudo_random_number_generator
from cell_fitting.optimization.bio_inspired import evaluators
from cell_fitting.optimization.fitter import FitterFactory
from cell_fitting.optimization.optimizer.optimizer_interface import Optimizer
from cell_fitting.util import merge_dicts


def mp_evaluator(candidate, args):  # top-level evaluator for multiprocessing
    if getattr(args['optimization_settings']['fitter_params'], 'mechanism_dir', None) is not None:
        args['optimization_settings']['fitter_params']['mechanism_dir'] = None
    fitter = FitterFactory().make_fitter(args['optimization_settings']['fitter_params'])
    evaluator = evaluators.create_evaluator(fitter.evaluate_fitness)
    if args['algorithm_settings']['normalize']:
        evaluator = evaluators.normalize_evaluator(evaluator,
                                                   args['optimization_settings']['bounds']['lower_bounds'],
                                                   args['optimization_settings']['bounds']['upper_bounds'])
    return evaluator(candidate, args)


class InspyredOptimizer(Optimizer):

    kind_of_algorithm = {'DEA': ['ec'], 'PSO': ['swarm'], 'SA': ['ec'], 'NSGA2': ['ec', 'emo']}

    def __init__(self, optimization_settings, algorithm_settings):
        super(InspyredOptimizer, self).__init__(optimization_settings, algorithm_settings)

        self.algorithm = self.get_algorithm()
        self.set_optimization_params()
        self.args = self.set_args()

        observer = observers.individuals_observer
        generator = functools.partial(getattr(generators, self.optimization_settings.generator),
                                      lower_bounds=self.optimization_settings.bounds['lower_bounds'],
                                      upper_bounds=self.optimization_settings.bounds['upper_bounds'])
        evaluator = evaluators.create_evaluator(self.optimization_settings.fitter.evaluate_fitness)
        terminator = getattr(inspyred.ec.terminators, self.optimization_settings.stop_criterion[0])

        if self.algorithm_settings.normalize:
            self.algorithm.observer = observers.normalize_observer(observer,
                                                                   self.optimization_settings.bounds['lower_bounds'],
                                                                   self.optimization_settings.bounds['upper_bounds'])
            self.generator = generators.normalize_generator(generator,
                                                            self.optimization_settings.bounds['lower_bounds'],
                                                            self.optimization_settings.bounds['upper_bounds'])
            self.evaluator = evaluators.normalize_evaluator(evaluator,
                                                            self.optimization_settings.bounds['lower_bounds'],
                                                            self.optimization_settings.bounds['upper_bounds'])
            self.bounder = inspyred.ec.Bounder(0, 1)
        else:
            self.algorithm.observer = observer
            self.generator = generator
            self.bounder = inspyred.ec.Bounder(self.optimization_settings.bounds['lower_bounds'],
                                               self.optimization_settings.bounds['upper_bounds'])
            self.evaluator = evaluator
        self.algorithm.terminator = terminator

    def get_algorithm(self):
        algorithm_name = self.algorithm_settings.algorithm_name
        algorithm_kind = InspyredOptimizer.kind_of_algorithm[algorithm_name]
        random = create_pseudo_random_number_generator(self.optimization_settings.seed)
        return getattr(reduce(getattr, [inspyred] + algorithm_kind), algorithm_name)(random)

    def set_optimization_params(self):
        if self.algorithm_settings.optimization_params is not None:
            variator_names = self.algorithm_settings.optimization_params.get('variator')
            if variator_names is not None:
                variators = [getattr(inspyred.ec.variators, name) for name in variator_names]
                self.algorithm.variator = variators

    def set_args(self):
        args = dict()
        args['individuals_file'] = open(os.path.join(self.algorithm_settings.save_dir, 'candidates.csv'), 'w')
        if self.optimization_settings.stop_criterion[0] == 'generation_termination':
            args['max_generations'] = self.optimization_settings.stop_criterion[1]
        args = merge_dicts(args, self.algorithm_settings.algorithm_params)
        args['optimization_settings'] = self.optimization_settings.to_dict()  # for multiprocessing
        args['algorithm_settings'] = self.algorithm_settings.to_dict()  # for multiprocessing
        if self.optimization_settings.extra_args.get('init_candidates', None):
            args['init_candidates'] = iter(self.generate_initial_candidates())
            self.optimization_settings.generator = 'init_candidates_generator'
        return args

    def optimize(self):
        self.algorithm.evolve(generator=self.generator,
                              evaluator=self.evaluator,  # if not using multiprocessing
                              #evaluator=inspyred.ec.evaluators.parallel_evaluation_mp,  # for multiprocessing
                              #mp_evaluator=mp_evaluator,  # for multiprocessing (must be pickable)
                              pop_size=self.optimization_settings.n_candidates,
                              maximize=self.optimization_settings.maximize,
                              bounder=self.bounder,
                              **self.args)


class SimulatedAnnealingOptimizer(InspyredOptimizer):

    def __init__(self, optimization_settings, algorithm_settings):
        super(SimulatedAnnealingOptimizer, self).__init__(optimization_settings, algorithm_settings)

        if self.algorithm_settings.normalize:
            self.algorithm.observer = observers.normalize_observer(observers.collect_observer,
                                                                   self.optimization_settings.bounds['lower_bounds'],
                                                                   self.optimization_settings.bounds['upper_bounds'])
        else:
            self.algorithm.observer = observers.collect_observer
        self.args['individuals'] = list()

    def optimize(self):
        random = create_pseudo_random_number_generator(self.optimization_settings.seed)

        for id in range(self.optimization_settings.n_candidates):
            self.args['id'] = id
            self.algorithm.evolve(generator=self.generator,
                                  evaluator=self.evaluator,
                                  pop_size=self.optimization_settings.n_candidates,
                                  maximize=self.optimization_settings.maximize,
                                  bounder=self.bounder,
                                  seeds=[self.generator(random, args=self.args)],  #TODO: self.args = None
                                  **self.args)

        self.save_candidates()

    def save_candidates(self):
        individuals_data = pd.DataFrame(self.args['individuals'], columns=['generation', 'id', 'fitness', 'candidate'])
        individuals_data = individuals_data.groupby('generation').apply(lambda x: x.sort_values(['fitness']))
        individuals_data.to_csv(self.args['individuals_file'], header=True, index=False)
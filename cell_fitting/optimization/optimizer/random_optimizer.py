import pandas as pd
import os
from cell_fitting.optimization.optimizer.optimizer_interface import Optimizer


class RandomOptimizer(Optimizer):

    def __init__(self, optimization_settings, algorithm_settings):
        super(RandomOptimizer, self).__init__(optimization_settings, algorithm_settings)
        self.initial_candidates = self.generate_initial_candidates()
        self.candidates = list()

        if self.optimization_settings.stop_criterion[0] == 'generation_termination':
            self.max_iterations = self.optimization_settings.stop_criterion[1] + 1
        else:
            raise ValueError('Only generation_termination implemented so far.')

    def optimize(self):
        for i in range(self.max_iterations):
            for id, candidate in enumerate(self.initial_candidates):
                self.store_candidates(candidate, i, id)
        self.save_candidates()

    def store_candidates(self, candidate, num_iterations, id):
        fitness = self.optimization_settings.fitter.evaluate_fitness(candidate, None)
        self.candidates.append([num_iterations, id, fitness,
                                str(candidate).replace(',', '').replace('[', '').replace(']', '')])

    def save_candidates(self):
        individuals_data = pd.DataFrame(self.candidates, columns=['generation', 'id', 'fitness', 'candidate'])
        individuals_data = individuals_data.groupby('generation').apply(lambda x: x.sort_values(['fitness']))
        with open(os.path.join(self.algorithm_settings.save_dir, 'candidates.csv'), 'w') as f:
            individuals_data.to_csv(f, header=True, index=False)
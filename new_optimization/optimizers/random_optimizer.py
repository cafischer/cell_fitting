import pandas as pd

from new_optimization import *
from optimization.bio_inspired import generators


class RandomOptimizer(Optimizer):

    def __init__(self, optimization_settings, algorithm_settings):
        super(RandomOptimizer, self).__init__(optimization_settings, algorithm_settings)
        self.candidates = list()

        if self.optimization_settings.stop_criterion[0] == 'generation_termination':
            self.max_iterations = self.optimization_settings.stop_criterion[1] + 1
        else:
            raise ValueError('Only generation_termination implemented so far.')

    def optimize(self):

        generator = getattr(generators, self.optimization_settings.generator)
        random = create_pseudo_random_number_generator(self.optimization_settings.seed)

        for i in range(self.max_iterations):
            for j in range(self.optimization_settings.n_candidates):
                candidate = generator(random, self.optimization_settings.bounds['lower_bounds'],
                                      self.optimization_settings.bounds['upper_bounds'], None)
                self.store_candidates(candidate, i, j)

        self.save_candidates()

    def store_candidates(self, candidate, num_iterations, id):
        fitness = self.optimization_settings.fitter.evaluate_fitness(candidate, None)
        self.candidates.append([num_iterations, id, fitness,
                                str(candidate).replace(',', '').replace('[', '').replace(']', '')])

    def save_candidates(self):
        individuals_data = pd.DataFrame(self.candidates, columns=['generation', 'id', 'fitness', 'candidate'])
        individuals_data = individuals_data.groupby('generation').apply(lambda x: x.sort_values(['fitness']))
        with open(self.algorithm_settings.save_dir + 'candidates.csv', 'w') as f:
            individuals_data.to_csv(f, header=True, index=False)
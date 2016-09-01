from inspyred import ec
from optimization.bio_inspired.inspyred_extension.generators import get_random_numbers_in_bounds
from optimization.bio_inspired.inspyred_extension.observers import individuals_observer


class Problem(object):

    def __init__(self, name, maximize, lower_bound, upper_bound):
        self.name = name
        self.maximize = maximize
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def generator(self, random, args):
        return get_random_numbers_in_bounds(random, self.lower_bound, self.upper_bound)

    def bounder(self, candidate, args):
        return ec.Bounder(self.lower_bound, self.upper_bound)(candidate, args)

    def observer(self,population, num_generations, num_evaluations, args):
        return individuals_observer(population, num_generations, num_evaluations, args)

    def evaluator(self, candidates, args):
        fitness = list()
        for candidate in candidates:
            fitness.append(self.evaluate(candidate, args))
        return fitness

    def evaluate(self, candidate, args):
        pass


class NormalizedProblem(Problem):
    def __init__(self, name, maximize, lower_bound, upper_bound, n_variables):
        super(NormalizedProblem, self).__init__(name, maximize, lower_bound, upper_bound)
        self.n_variables = n_variables

    def generator(self, random, args):
        return get_random_numbers_in_bounds(random, [0] * self.n_variables, [1] * self.n_variables)

    def bounder(self, candidate, args):
        return ec.Bounder(0, 1)(candidate, args)

    def observer(self,population, num_generations, num_evaluations, args):
        population_unnormed = NormalizedProblem.unnorm_population(population, self.lower_bound, self.upper_bound)
        return individuals_observer(population_unnormed, num_generations, num_evaluations, args)

    def evaluator(self, candidates, args):
        fitness = list()
        for candidate in candidates:
            fitness.append(self.evaluate(self.unnorm_candidate(candidate, self.lower_bound, self.upper_bound), args))
        return fitness

    def evaluate(self, candidate, args):
        pass

    @staticmethod
    def unnorm_candidate(candidate, lower_bound, upper_bound):
        return [candidate[i] * (upper_bound[i] - lower_bound[i]) + lower_bound[i] for i in range(len(candidate))]

    @staticmethod
    def norm_candidate(candidate, lower_bound, upper_bound):
        return [(candidate[i] - lower_bound[i]) / (upper_bound[i] - lower_bound[i]) for i in range(len(candidate))]

    @staticmethod
    def unnorm_population(population, lower_bound, upper_bound):
        population_unnormed = list()
        for i, p in enumerate(population):
            individual = ec.Individual()  # explicitly copy because candidate setter changes fitness to None
            individual.candidate = NormalizedProblem.unnorm_candidate(p.candidate, lower_bound, upper_bound)
            individual.fitness = p.fitness
            individual.birthdate = p.birthdate
            individual.maximize = p.maximize
            population_unnormed.append(individual)
        return population_unnormed

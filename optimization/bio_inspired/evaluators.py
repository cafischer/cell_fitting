from util import *


def create_evaluator(evaluate):

    def evaluator(candidates, args):
        fitness = list()
        for candidate in candidates:
            fitness.append(evaluate(candidate, args))
        return fitness

    return evaluator


def normalize_evaluator(evaluator, lower_bounds, upper_bounds):

    def evaluator_normalized(candidates, args):
        candidates = [unnorm(candidate, lower_bounds, upper_bounds) for candidate in candidates]
        return evaluator(candidates, args)

    return evaluator_normalized
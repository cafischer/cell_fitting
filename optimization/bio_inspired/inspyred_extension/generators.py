
__author__ = 'caro'


def randomNumberForEachLowerUpperBound(random, lower_bounds, upper_bounds):
    assert(len(lower_bounds) == len(upper_bounds))
    assert(all([lower_bounds[i] <= upper_bounds[i] for i in range(len(lower_bounds))]))
    return [random.uniform(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))]
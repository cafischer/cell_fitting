
__author__ = 'caro'


def get_random_numbers_in_bounds(random_number_generator, lower_bounds, upper_bounds):
    assert(len(lower_bounds) == len(upper_bounds))
    assert(all([lower_bounds[i] <= upper_bounds[i] for i in range(len(lower_bounds))]))
    return [random_number_generator.uniform(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))]
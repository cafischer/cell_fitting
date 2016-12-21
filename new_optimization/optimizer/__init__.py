from inspyred_optimizer import *
from scipy_optimizer import *
from random_optimizer import *


class OptimizerFactory:

    def __init__(self):
        pass

    def make_optimizer(self, optimization_settings, algorithm_settings):
        algorithm_name = algorithm_settings.algorithm_name
        if algorithm_name == 'SA':
            return SimulatedAnnealingOptimizer(optimization_settings, algorithm_settings)
        elif algorithm_name == 'PSO' or algorithm_name == 'DEA':
            return InspyredOptimizer(optimization_settings, algorithm_settings)
        elif algorithm_name == 'Nelder-Mead' or algorithm_name == 'Powell' or algorithm_name == 'CG'\
                or algorithm_name == 'BFGS' or algorithm_name == 'Newton-CG' or algorithm_name == 'L-BFGS-B'\
                or algorithm_name == 'TNC' or algorithm_name == 'SLSQP'\
                or algorithm_name == 'dogleg' or algorithm_name == 'trust-ncg':
            return ScipyOptimizer(optimization_settings, algorithm_settings)
        elif algorithm_name == 'Random':
            return RandomOptimizer(optimization_settings, algorithm_settings)
        else:
            raise ValueError('Optimizer for ' + algorithm_name + ' does not exist!')
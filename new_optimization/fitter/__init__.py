import abc
from hodgkinhuxleyfitter import *
from izhikevichfitter import *
from linearregressionfitter import *

class Fitter:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def evaluate_fitness(self, candidate):
        pass

    @abc.abstractmethod
    def __dict__(self):
        pass


class FitterFactory:

    def __init__(self):
        pass

    def make_fitter(self, fitter_params):
        name = fitter_params['name']
        if name == 'HodgkinHuxleyFitter':
            return HodgkinHuxleyFitter(**fitter_params)
        elif name == 'HodgkinHuxleyFitterSeveralData':
            return HodgkinHuxleyFitterSeveralData(**fitter_params)
        elif name == 'HodgkinHuxleyFitterPareto':
            return HodgkinHuxleyFitterPareto(**fitter_params)
        else:
            raise ValueError('Fitter '+name+' not available!')
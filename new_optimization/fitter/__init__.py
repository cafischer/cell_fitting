import abc


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
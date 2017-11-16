import abc


class Fitter:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def evaluate_fitness(self, candidate, args):
        pass

    @abc.abstractmethod
    def to_dict(self):
        pass
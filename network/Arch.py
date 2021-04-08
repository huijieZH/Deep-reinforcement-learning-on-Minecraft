from abc import ABCMeta, abstractmethod

class MineCraftRL(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, args):
        pass

    @abstractmethod
    def train(self):
        pass

    
    @abstractmethod
    def step_train(self):
        pass
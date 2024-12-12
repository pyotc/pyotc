from abc import ABC, abstractmethod

class OTC(ABC):
    def __init__(self, P, Q, C):
        self.P = P
        self.Q = Q
        self.C = C
        self.R = self.reset()

    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self):
        pass
    

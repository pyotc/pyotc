import numpy as np
from numpy.linalg import pinv
from scipy.optimize import linprog
from scipy.linalg import fractional_matrix_power
import copy

class OTC:
    def __init__(self, P: np.ndarray, Q: np.ndarray, c: np.ndarray) -> None:
        self.P = P
        self.Q = Q
        self.c = c
        self.dx = P.shape[0]
        self.dy = Q.shape[0]
        self.R = np.ones((self.dx*self.dy, self.dx*self.dy))
    
    def independent_transition_coupling(self):
        """
        Compute the independent transition coupling.
        """
        raise NotImplementedError
    
    def evaluate(self):
        """
        Evaluate transition coupling.
        """
        raise NotImplementedError

    def improve(self):
        """
        Improve transition coupling
        """
        raise NotImplementedError

class ExactOTC(OTC);
    raise NotImplementedError
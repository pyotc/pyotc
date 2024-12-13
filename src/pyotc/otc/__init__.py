from abc import ABC, abstractmethod
import numpy as np
import itertools

class OTC(ABC):
    def __init__(self, P: list[np.ndarray], Q: list[np.ndarray], c: np.ndarray) -> None:
        """
        Initialize the basic OTC object.

        TODO: specify precision (yuning issue)
        Args:
            P list[np.ndarray]: list of transition matrices
            Q list[np.ndarray]: list of tranistion matrices
            c list[np.ndarray]: list of cost matrices (one for each coupling)
        """
        assert len(P)*len(Q) == len(c)
        self.P = P
        self.Q = Q
        self.c = c
        self.R = self.initial_couplings()

    @staticmethod
    def independent_transition_coupling(Px, Py):
        """
        Compute the independent transition coupling.
        #TODO: confirm that this is just np.kron and replace.
        """
        dx, dx_col = Px.shape
        dy, dy_col = Py.shape

        P_ind = np.zeros((dx*dy, dx_col*dy_col))
        for x_row in range(dx):
            for x_col in range(dx_col):
                for y_row in range(dy):
                    for y_col in range(dy_col):
                        idx1 = dy*(x_row) + y_row
                        idx2 = dy*(x_col) + y_col
                        P_ind[idx1, idx2] = Px[x_row, x_col] * Py[y_row, y_col]
        return P_ind

    def reset(self)->list[np.ndarray]:
        """Reset to original product coupling P_i \otimes Q_j

        Returns:
            list[np.ndarray]: P_i \otimes Q_j
        """
        prod = itertools.product(self.P, self.Q)
        return [self.independent_transition_coupling(*pair) for pair in prod]
    
    @abstractmethod
    def step(self):
        """
        Compute one step of an OTC algorithm.

        This is a *step* and not necessarily an *improvement* we leave it to the API developer
        to break up evaluation or improvement.
        """
        raise NotImplementedError    

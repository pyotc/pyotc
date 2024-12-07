"""Yuning's other other native implementation of lp ot"""
import numpy as np
from scipy.optimize import linprog

class OT:
    def __init__(self, cost, r, c) -> None:
        self.cost = cost.reshape(-1, 1)
        self.nx = r.size
        self.ny = c.size
        self.A_eq = self._build_A_eq()
        self.b_eq = self._build_b_eq(r, c)
        self.bound = [[0, None]] * (self.nx*self.ny)
    
    def _build_A_eq(self):
        Aeq = np.zeros((self.nx+self.ny, self.nx*self.ny))
        # column sums correct
        for row in range(self.nx):
            for t in range(self.ny):
                Aeq[row, (row*self.ny)+t] = 1

        # row sums correct
        for row in range(self.nx, self.nx+self.ny):
            for t in range(self.nx):
                Aeq[row, t*self.ny+(row-self.nx)] = 1
        
        return Aeq
    
    def _build_b_eq(self, r, c):
        beq = np.concatenate((r.flatten(), c.flatten()))
        return beq.reshape(-1,1)
        
    def __call__(self) -> np.Any:
        res = linprog(self.cost, A_eq=self.A_eq, b_eq=self.b_eq, 
                      bounds=self.bound, method='highs-ipm')
        return res.x, res.fun

def compute_ot(C: np.ndarray, r: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        C (np.ndarray): cost vector
        r (np.ndarray): _description_
        c (np.ndarray): _description_

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    lp_ot = OT(cost=C, r=r, c=c)
    lp_sol, lp_val = lp_ot()
    return lp_sol, lp_val


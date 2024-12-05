import numpy as np
from numpy.linalg import pinv
from scipy.optimize import linprog
from scipy.linalg import fractional_matrix_power
import copy
import itertools

class OTC:
    def __init__(self, P: list[np.ndarray], Q: list[np.ndarray], c: np.ndarray) -> None:
        """
        Initialize the basic OTC object.

        TODO: specify precision (yuning issue)
        TODO: calculating the stationary version with eigenvalues
        Args:
            P list[np.ndarray]: list of transition matrices
            Q list[np.ndarray]: list of tranistion matrices
            c list[np.ndarray]: list of cost matrices (one for each coupling)
        """
        assert len(P)*len(Q) == len(c)
        self.P = P
        self.Q = Q
        self.c = c
        self.R = self.reset()

    def initial_couplings(self)->list[np.ndarray]:
        prod = itertools.product(self.P, self.Q)
        return [self.independent_transition_coupling(*pair) for pair in prod]

    @staticmethod
    def independent_transition_coupling(Px, Py):
        """
        Compute the independent transition coupling.
        #TODO: confirm that this is just np.kron and replace.
        """
        dx, dx_col = Px.shape # TODO: this pattern is different
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
    
    def evaluate(self):
        """
        Evaluate transition coupling.
        """
        raise NotImplementedError
    
    def step(self):
        """
        Compute one step of an OTC algorithm.

        This is a *step* and not necessarily an *improvement*
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset to initial coupling

        Returns:
            _type_: _description_
        """
        raise NotImplementedError

class ExactOTC(OTC):
    def evaluate(self):
        """Evaluate the coupling.

        Returns:
            _type_: _description_
        """
        d = self.R.shape[0]
        c_flat = np.reshape(self.c, (d, -1))
        A = np.block([[np.eye(d) - self.R, np.zeros((d, d)), np.zeros((d, d))],
                  [np.eye(d), np.eye(d) - self.R, np.zeros((d, d))],
                  [np.zeros((d, d)), np.eye(d), np.eye(d) - self.R]])
        b = np.concatenate([np.zeros((d, 1)), c_flat, np.zeros((d, 1))])
        try:
            sol = np.linalg.solve(A, b)
        except: 
            sol = np.matmul(pinv(A), b)

        g = sol[0:d].flatten()
        h = sol[d:2*d].flatten()
        return g, h

    def step(self, g: np.ndarray, h: np.ndarray)->None:
        """Perform the exact step proposed in original OTC work.

        Args:
            g (np.ndarray): _description_
            h (np.ndarray): _description_

        Returns:
            None: updates self.R
        """
        Pz = np.zeros((self.dx*self.dy, self.dx*self.dy))
        g_const = True
        for i in range(self.dx):
            for j in range(i+1, self.dx):
                if abs(g[i] - g[j]) > 1e-3:
                    g_const = False
                    break
            if not g_const:
                break
        # If g is not constant, improve transition coupling against g.
        if not g_const:
            g_mat = np.reshape(g, (dx, dy))
            for x_row in range(dx):
                for y_row in range(dy):
                    dist_x = Px[x_row, :]
                    dist_y = Py[y_row, :]
                    # Check if either distribution is degenerate.
                    if any(dist_x == 1) or any(dist_y == 1):
                        sol = np.outer(dist_x, dist_y)
                    # If not degenerate, proceed with OT.
                    else:
                        sol, val = computeot_lp(g_mat, dist_x, dist_y)
                    idx = dy*(x_row)+y_row
                    Pz[idx, :] = np.reshape(sol, (-1, dx*dy))
            return Pz
        ## Try to improve with respect to h.
        h_mat = np.reshape(h, (dx, dy))
        for x_row in range(dx):
            for y_row in range(dy):
                dist_x = Px[x_row, :]
                dist_y = Py[y_row, :]
                # Check if either distribution is degenerate.
                if any(dist_x == 1) or any(dist_y == 1):
                    sol = np.outer(dist_x, dist_y)
                # If not degenerate, proceed with OT.
                else:
                    sol, val = computeot_lp(h_mat, dist_x, dist_y)
                idx = dy*(x_row)+y_row
                Pz[idx, :] = np.reshape(sol, (-1, dx*dy))
        self.R = Pz
        
        

    def improve(self, g, h):
        """Origianl improve implementation.

        Args:
            g (_type_): _description_
            h (_type_): _description_

        Returns:
            _type_: _description_
        """
        Pz = np.zeros((self.dx*self.dy, self.dx*self.dy))
        g_const = True
        for i in range(self.dx):
            for j in range(i+1, self.dx):
                if abs(g[i] - g[j]) > 1e-3:
                    g_const = False
                    break
            if not g_const:
                break
        # If g is not constant, improve transition coupling against g.
        if not g_const:
            g_mat = np.reshape(g, (dx, dy))
            for x_row in range(dx):
                for y_row in range(dy):
                    dist_x = Px[x_row, :]
                    dist_y = Py[y_row, :]
                    # Check if either distribution is degenerate.
                    if any(dist_x == 1) or any(dist_y == 1):
                        sol = np.outer(dist_x, dist_y)
                    # If not degenerate, proceed with OT.
                    else:
                        sol, val = computeot_lp(g_mat, dist_x, dist_y)
                    idx = dy*(x_row)+y_row
                    Pz[idx, :] = np.reshape(sol, (-1, dx*dy))
            return Pz
        ## Try to improve with respect to h.
        h_mat = np.reshape(h, (dx, dy))
        for x_row in range(dx):
            for y_row in range(dy):
                dist_x = Px[x_row, :]
                dist_y = Py[y_row, :]
                # Check if either distribution is degenerate.
                if any(dist_x == 1) or any(dist_y == 1):
                    sol = np.outer(dist_x, dist_y)
                # If not degenerate, proceed with OT.
                else:
                    sol, val = computeot_lp(h_mat, dist_x, dist_y)
                idx = dy*(x_row)+y_row
                Pz[idx, :] = np.reshape(sol, (-1, dx*dy))
        return Pz


class EntropicOTC(OTC):
    def evaluate(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the ApproxTCE (Algorithm 2a)

        Returns:
            tuple[np.ndarray, np.ndarray]: _description_
        """

    def step(self, h) -> None:
        """Compute the EntropicTCI (Algorithm 2b)

        Args:
            h (_type_): _description_
        """
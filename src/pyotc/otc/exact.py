from . import OTC
import numpy as np

class ExactOTC(OTC):
    #  TODO: calculating the stationary version with eigenvalues
    @staticmethod
    def evaluate_individual(R, c) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate an individual coupling R given cost c.
        """
        d = R.shape[0]
        c_flat = np.reshape(c, (d, -1))
        A = np.block([[np.eye(d) - R, np.zeros((d, d)), np.zeros((d, d))],
                  [np.eye(d), np.eye(d) - R, np.zeros((d, d))],
                  [np.zeros((d, d)), np.eye(d), np.eye(d) - R]])
        b = np.concatenate([np.zeros((d, 1)), c_flat, np.zeros((d, 1))])
        try:
            sol = np.linalg.solve(A, b)
        except: 
            sol = np.matmul(pinv(A), b)

        g = sol[0:d].flatten()
        h = sol[d:2*d].flatten()
        return g, h

    
    def evaluate(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Evaluate implementation interface for ExactOTC.

        Returns:
            list[tuple[np.ndarray, np.ndarray]]: _description_
        """
        return [self.evaluate_individual(r,c) for r,c in zip(self.R, self.c)]
    

    @staticmethod
    def improve_individual(g: np.ndarray, h: np.ndarray)->np.ndarray:
        """Perform the exact step proposed in original OTC work.

        Args:
            g (np.ndarray): from evaluate
            h (np.ndarray): from evaluate

        Returns:
            R (np.ndarray)
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
    
    def improve(self, evaluations: list[tuple[np.ndarray, np.ndarray]])->list[np.ndarray]:
        """Improve across all couplings in R.

        Returns:
            list[np.ndarray]: improved R
        """
        return [self.improve_individual(g=g, h=h) for g, h in evaluations]
    
    def step(self):
        evaluations = self.evaluate()
        self.R = self.improve(evaluations=evaluations)
        return None
    
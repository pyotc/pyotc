"""
Original Transition Coupling Improvements (TCI) methods from:
https://jmlr.csail.mit.edu/papers/volume23/21-0519/21-0519.pdf
"""
import numpy as np
import pyotc.otc.backend.optimal_transport as ot_backend
import copy

class ExactTCI:
    def __init__(self, Px, Py, ot_method="scipy") -> None:
        self.Px = Px
        self.Py = Py
        self.dx = Px.shape[0]
        self.dy = Py.shape[0]
        self.backend = getattr(ot_backend, ot_method)

    def _test_constant(self, g, tol=1e-3):
        g_const = True
        for i in range(self.dx):
            for j in range(i+1, self.dx):
                if abs(g[i] - g[j]) > tol:
                    g_const = False
                    break
            if not g_const:
                break
        return g_const

    def _improve(self, f):
        Pz = np.zeros((self.dx*self.dy, self.dx*self.dy))
        f_mat = np.reshape(f, (self.dx, self.dy))
        for x_row in range(self.dx):
            for y_row in range(self.dy):
                dist_x = self.Px[x_row, :]
                dist_y = self.Py[y_row, :]
                # Check if either distribution is degenerate.
                if any(dist_x == 1) or any(dist_y == 1):
                    sol = np.outer(dist_x, dist_y)
                # If not degenerate, proceed with OT.
                else:
                    sol, val = self.backend.compute_ot(f_mat, dist_x, dist_y)
                idx = self.dy*(x_row)+y_row
                Pz[idx, :] = np.reshape(sol, (-1, self.dx*self.dy))
        return Pz
        
    def __call__(self, g, h):
        # improve coupling
        if not self._test_constant(g=g):
            return self._improve(f=g)
        else:
            return self._improve(f=h)


# TODO: document unit test, replace print with logging
# TODO: perhaps break up further
# TODO: use the class above to replace all but the acceptance metric
def exact_tci(g, h, P0, Px, Py, ot_method="scipy"):
    # dynamically load backend for transport method
    backend = getattr(ot_backend, ot_method)

    # set shapes
    dx = Px.shape[0]
    dy = Py.shape[0]
    Pz = np.zeros((dx*dy, dx*dy))
    
    g_const = True
    for i in range(dx):
        for j in range(i+1, dx):
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
                    sol, val = backend.compute_ot(g_mat, dist_x, dist_y)
                idx = dy*(x_row)+y_row
                Pz[idx, :] = np.reshape(sol, (-1, dx*dy))
                #P[idx, :] = sol
        if np.max(np.abs(np.matmul(P0, g) - np.matmul(Pz, g))) <= 1e-7:
            Pz = copy.deepcopy(P0)
        else:
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
                sol, val = backend.compute_ot(h_mat, dist_x, dist_y)
            idx = dy*(x_row)+y_row
            # print(x_row, y_row, P0[18, 1])
            Pz[idx, :] = np.reshape(sol, (-1, dx*dy))

    if np.max(np.abs(np.matmul(P0, h) - np.matmul(Pz, h))) <= 1e-4:
        Pz = copy.deepcopy(P0)
    return Pz

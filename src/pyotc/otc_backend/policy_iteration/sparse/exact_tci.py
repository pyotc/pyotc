import numpy as np
import scipy.sparse as sp
from pyotc.otc_backend.optimal_transport.pot import computeot_pot

def setup_ot(f, Px, Py, Pz):
    dx = Px.shape[0]
    dy = Py.shape[0]
    f_mat = np.reshape(f, (dx, dy))

    for x_row in range(dx):
        for y_row in range(dy):
            dist_x = Px[x_row, :]
            dist_y = Py[y_row, :]
            # degenerate distribution check
            if np.any(dist_x == 1) or np.any(dist_y == 1):
                sol = np.outer(dist_x, dist_y)
            else:
                sol, _ = computeot_pot(f_mat, dist_x, dist_y) 
            idx = dy * x_row + y_row
            sol_flat = sol.flatten()
            for j in np.nonzero(sol_flat)[0]:
                Pz[idx, j] = sol_flat[j]
    return Pz

def exact_tci(g, h, P0, Px, Py): 
    dx, dy = Px.shape[0], Py.shape[0]
    Pz = sp.lil_matrix((dx * dy, dx * dy))
    g_const = np.max(g) - np.min(g) <= 1e-3

    if not g_const:
        Pz = setup_ot(g, Px, Py, Pz) 
        if np.max(np.abs(P0.dot(g) - Pz.dot(g))) <= 1e-7:
            Pz = P0.copy()
        else:
            return Pz

    Pz = setup_ot(h, Px, Py, Pz) 
    if np.max(np.abs(P0.dot(h) - Pz.dot(h))) <= 1e-4:
        Pz = P0.copy()
    return Pz

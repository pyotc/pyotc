import numpy as np
import copy
from pyotc.otc_backend.optimal_transport.pot import computeot_pot

def exact_tci(g, h, P0, Px, Py):
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
                    sol, val = computeot_pot(g_mat, dist_x, dist_y)
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
                sol, val = computeot_pot(h_mat, dist_x, dist_y)
            idx = dy*(x_row)+y_row
            # print(x_row, y_row, P0[18, 1])
            Pz[idx, :] = np.reshape(sol, (-1, dx*dy))

    if np.max(np.abs(np.matmul(P0, h) - np.matmul(Pz, h))) <= 1e-4:
        Pz = copy.deepcopy(P0)
    return Pz
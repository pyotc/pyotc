"""
Original Transition Coupling Improvements (TCI) methods from:
https://jmlr.csail.mit.edu/papers/volume23/21-0519/21-0519.pdf
"""
import numpy as np
from pyotc.otc_backend.optimal_transport.native import computeot_lp


# TODO: document unit test, replace print with logging
# TODO: perhaps break up further
def exact_tci(g, h, P0, Px, Py):
    #x_sizes = Px.shape
    #y_sizes = Py.shape
    #dx = x_sizes[0]
    #dy = y_sizes[0]
    dx = Px.shape[0]
    dy = Py.shape[0]
    Pz = np.zeros((dx*dy, dx*dy))
    #print(1, P0[18, 1])
    ## Try to improve with respect to g.
    # Check if g is constant.
    g_const = True
    for i in range(dx):
        for j in range(i+1, dx):
            if abs(g[i] - g[j]) > 1e-3:
                g_const = False
                break
        if not g_const:
            break
    # If g is not constant, improve transition coupling against g.
    #print(2, P0[18, 1])
    #print(P[18, 1])
    if not g_const:
        #g_mat = np.reshape(g, (dx, dy)).T
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
                #P[idx, :] = sol
        if np.max(np.abs(np.matmul(P0, g) - np.matmul(Pz, g))) <= 1e-7:
            #print('HERER')
            Pz = copy.deepcopy(P0)
        else:
            return Pz
    #print(3, P0[18, 1])
    #print(P[18, 1])
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
            # print(x_row, y_row, P0[18, 1])
            Pz[idx, :] = np.reshape(sol, (-1, dx*dy))

    if np.max(np.abs(np.matmul(P0, h) - np.matmul(Pz, h))) <= 1e-4:
        Pz = copy.deepcopy(P0)
        #print('12312312')
    #print(4, P0[18, 1])
    return Pz
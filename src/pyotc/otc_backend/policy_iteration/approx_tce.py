import numpy as np


def approx_tce(P, c, L, T):
    d = P.shape[0]
    c = np.reshape(c, (d, -1))
    c_max = np.max(c)

    g_old = c
    g = P @ g_old
    l = 1
    tol = 1e-12
    while l <= L and np.max(np.abs(g - g_old)) > tol * c_max:
        g_old = g
        g = P @ g_old
        l += 1

    g = np.mean(g) * np.ones((d, 1))
    diff = c - g
    h = diff.copy()
    t = 1
    while t <= T and np.max(np.abs(P @ diff)) > tol * c_max:
        h += P @ diff
        diff = P @ diff
        t += 1

    return g, h

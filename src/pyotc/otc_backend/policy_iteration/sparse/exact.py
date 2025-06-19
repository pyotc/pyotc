import numpy as np
import scipy.sparse as sp

from .exact_tce import exact_tce
from .exact_tci import exact_tci
from ..utils import get_stat_dist
import time

def exact_otc(Px, Py, c, stat_dist='best', max_iter=100):
    
    start = time.time()
    print("Starting exact_otc_sparse...")
    dx, dy = Px.shape[0], Py.shape[0]
    P = sp.kron(sp.csr_matrix(Px), sp.csr_matrix(Py), format='csr')
    
    for iter in range(max_iter):
        print("Iteration:", iter)
        P_old = P.copy()
        
        print("Computing exact TCE...")
        g, h = exact_tce(P, c)
        
        print("Computing exact TCI...")
        P = exact_tci(g, h, P_old, Px, Py)

        if (P != P_old).nnz == 0:
            print("Convergence reached in {iter} iterations. Computing stationary distribution...")
            stat_dist = get_stat_dist(P, method=stat_dist, c=c)
            stat_dist = np.reshape(stat_dist, (dx, dy))
            exp_cost = g[0].item()
            end = time.time()
            print(f"[exact_otc] Finished. Total time elapsed: {end - start:.3f} seconds.")
            return float(exp_cost), P, stat_dist

    return None, None, None

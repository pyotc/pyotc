import numpy as np
import scipy.sparse as sp
import time

from .exact_tce import exact_tce
from .exact_tci import exact_tci
from ..utils import get_stat_dist


def exact_otc(Px, Py, c, stat_dist="best", max_iter=100):
    """
    Computes the optimal transport coupling (OTC) between two stationary Markov chains represented by transition matrices Px and Py,
    as described in Algorithm 1 of the paper: "Optimal Transport for Stationary Markov Chains via Policy Iteration"
    (https://www.jmlr.org/papers/volume23/21-0519/21-0519.pdf).

    The algorithm iteratively updates the transition coupling matrix until convergence by alternating
    between Transition Coupling Evaluation (TCE) and Transition Coupling Improvement (TCI) steps.

    For a detailed discussion of the connection between the OTC problem and Markov Decision Processes (MDPs), see Section 4 of the paper.
    Additional background on policy iteration methods for solving average-cost MDP problems can be found in Chapters 8 and 9 of
    "Markov Decision Processes: Discrete Stochastic Dynamic Programming" by Martin L. Puterman.

    Note:
        In the TCE step (implemented in exact_tce), we solve a block linear system using functions from scipy.sparse.linalg.
        However, when A in Ax = b is nearly singular, we have observed a few cases where both SciPy solvers (scipy.sparse.linalg.spsolve, scipy.sparse.linalg.lsmr)
        can produce results that differ from NumPy's solver (np.linalg.solve). This leads to discrepancies with the dense implementation and non-convergence.
        This is an issue with SciPy's sparse solvers and remains unresolved. The best approach in such cases is to use the dense implementation.

    Args:
        Px (np.ndarray): Transition matrix of the source Markov chain of shape (dx, dx).
        Py (np.ndarray): Transition matrix of the target Markov chain of shape (dy, dy).
        c (np.ndarray): Cost function of shape (dx, dy).
        stat_dist (str, optional): Method to compute the stationary distribution.
                                   Options include 'best', 'eigen', 'iterative' and None. Defaults to 'best'.
        max_iter (int, optional): Maximum number of iterations for the convergence process. Defaults to 100.

    Returns:
        exp_cost (float): Expected transport cost under the optimal transition coupling.
        R (scipy.sparse.csr_matrix): Optimal transition coupling matrix of shape (dx*dy, dx*dy).
        stat_dist (np.ndarray): Stationary distribution of the optimal transition coupling of shape (dx, dy).

        If convergence is not reached within max_iter iterations, returns (None, None, None).
    """

    start = time.time()
    print("Starting exact_otc_sparse...")
    dx, dy = Px.shape[0], Py.shape[0]

    # Initial coupling matrix using Kronecker product
    R = sp.kron(sp.csr_matrix(Px), sp.csr_matrix(Py), format="csr")

    for iter in range(max_iter):
        print("Iteration:", iter)
        R_old = R.copy()

        print("Computing exact TCE...")
        g, h = exact_tce(R, c)

        print("Computing exact TCI...")
        R = exact_tci(g, h, R_old, Px, Py)

        # Check if the transition coupling matrix has converged
        if (R != R_old).nnz == 0:
            if stat_dist is None:
                print(
                    f"Convergence reached in {iter + 1} iterations. No stationary distribution computation requested."
                )
                exp_cost = g[0].item()
                end = time.time()
                print(
                    f"[exact_otc] Finished. Total time elapsed: {end - start:.3f} seconds."
                )
                return float(exp_cost), R, None
            else:
                print(
                    f"Convergence reached in {iter + 1} iterations. Computing stationary distribution..."
                )
                stat_dist = get_stat_dist(R, method=stat_dist, c=c)
                stat_dist = np.reshape(stat_dist, (dx, dy))
                exp_cost = g[0].item()
                end = time.time()
                print(
                    f"[exact_otc] Finished. Total time elapsed: {end - start:.3f} seconds."
                )
                return float(exp_cost), R, stat_dist

    # Return None if convergence is not achieved
    print(f"Convergence not achieved after {iter} iterations. Returning None.")
    return None, None, None

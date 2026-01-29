"""
Entropic Optimal Transition Coupling (OTC) solvers.

Implements variants of the OTC algorithm using entropic regularization.
Includes both a custom Sinkhorn implementation and one based on the POT library.

References:
    - Section 5, "Optimal Transport for Stationary Markov Chains via Policy Iteration"
      (https://www.jmlr.org/papers/volume23/21-0519/21-0519.pdf)

Methods:
    - logsinkhorn: A self-implemented log-scaled Sinkhorn solver.
    - ot_sinkhorn: Sinkhorn solver from POT library.
    (reference: https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.sinkhorn)
    - ot_logsinkhorn: Sinkhorn solver from POT library in log scale.
    (reference: https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.sinkhorn_log)
    - ot_greenkhorn: Sinkhorn solver of greedy version from POT library.
    (reference: https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.greenkhorn)
"""

import time
import numpy as np
import ot
from ..utils import get_best_stat_dist
from .approx_tce import approx_tce
from .entropic_tci import entropic_tci
from pyotc.otc_backend.optimal_transport.logsinkhorn import logsinkhorn


def entropic_otc(
    Px,
    Py,
    c,
    L=100,
    T=100,
    xi=0.1,
    method="logsinkhorn",
    sink_iter=100,
    reg_num=None,
    get_sd=False,
    silent=True,
):
    """
    Solves the Entropic Optimal Transition Coupling (OTC) problem between two Markov chains
    using approximate policy iteration and entropic regularization.

    This method alternates between approximate coupling evaluation
    and entropic coupling improvement (via Sinkhorn iterations), until convergence.

    Args:
        Px (np.ndarray): Transition matrix of the source Markov chain of shape (dx, dx).
        Py (np.ndarray): Transition matrix of the target Markov chain of shape (dy, dy).
        c (np.ndarray): Cost function of shape (dx, dy).
        L (int): Number of iterations for computing the cost vector g in approx_tce.
        T (int): Number of iterations for computing the bias vector h in approx_tce.
        xi (float): Scaling factor for entropic cost adjustment in entropic_tci.
        method (str): Method for the Sinkhorn algorithm. Must choose from ['logsinkhorn', 'ot_sinkhorn', 'ot_logsinkhorn', 'ot_greenkhorn']. Default is 'logsinkhorn'. See 'Methods' above for details.
        sink_iter (int): Number of iterations for 'logsinkhorn' method. Maximum number of Sinkhorn iterations for other methods from POT library. Used in the entropic TCI step.
        reg_num (float): Entropic regularization term, used only for methods from POT package.
        get_sd (bool): If True, compute best stationary distribution using linear programming.
        silent (bool): If False, print convergence info during iterations and running time

    Returns:
        exp_cost (float): Expected transport cost under the optimal transition coupling.
        P (np.ndarray): Optimal transition coupling matrix of shape (dx*dy, dx*dy).
        stat_dist (Optional[np.ndarray]): Stationary distribution of the optimal transition coupling of shape (dx, dy),
                                            or None if get_sd is False.
    """
    if not silent:
        start_time = time.time()
        print(f"Starting entropic otc with {method} method...")

    dx, dy = Px.shape[0], Py.shape[0]
    max_c = np.max(c)
    tol = 1e-5 * max_c

    g_old = max_c * np.ones(dx * dy)
    g = g_old - 10 * tol
    P = np.kron(Px, Py)

    if method == "logsinkhorn":

        def solver_fn(A, a, b):
            return logsinkhorn(A, a, b, sink_iter)
    elif method == "ot_sinkhorn":
        if reg_num is None:
            raise ValueError("reg_num must be specified for 'ot_sinkhorn'")

        def solver_fn(A, a, b):
            return ot.sinkhorn(a, b, A, reg=reg_num, numItermax=sink_iter)
    elif method == "ot_logsinkhorn":
        if reg_num is None:
            raise ValueError("reg_num must be specified for 'ot_logsinkhorn'")

        def solver_fn(A, a, b):
            return ot.bregman.sinkhorn_log(a, b, A, reg=reg_num, numItermax=sink_iter)
    elif method == "ot_greenkhorn":
        if reg_num is None:
            raise ValueError("reg_num must be specified for 'ot_greenkhorn'")

        def solver_fn(A, a, b):
            return ot.bregman.greenkhorn(a, b, A, reg=reg_num, numItermax=sink_iter)
    else:
        raise ValueError(f"Unknown method: {method}")

    iter_ctr = 0
    while g_old[0] - g[0] > tol:
        iter_ctr += 1
        P_old = P
        g_old = g
        if not silent:
            print("Iteration:", iter_ctr)
            start_iter = time.time()

        # Approximate transition coupling evaluation
        if not silent:
            print("Computing entropic TCE...")
        g, h = approx_tce(P, c, L, T)

        # Entropic transition coupling improvement (passing solver function to entropic_tci)
        if not silent:
            print("Computing entropic TCE...")
        P = entropic_tci(h=h, P0=P_old, Px=Px, Py=Py, xi=xi, solver_fn=solver_fn)

        if not silent:
            iter_time = time.time() - start_iter
            elapsed = time.time() - start_time
            g0 = float(np.ravel(g)[0])
            g0_old = float(np.ravel(g_old)[0])
            diff = g0_old - g0
            ratio = diff / g0 if g0 != 0 else float("inf")
            print(
                f"[Iter {iter_ctr} taking {iter_time:.2f}s] Δg={diff:.3e}, g[0]={g0:.6f}, Δg/g[0]={ratio:.3e}, total elapsed={elapsed:.2f}s"
            )

    # In case of numerical instability, make non-negative and normalize.
    P = np.maximum(P, 0)
    row_sums = np.sum(P, axis=1, keepdims=True)
    P = P / np.where(row_sums > 0, row_sums, 1)

    if get_sd:
        if not silent:
            print(
                f"Convergence reached in {iter_ctr} iterations. Computing stationary distribution..."
            )
        stat_dist, exp_cost = get_best_stat_dist(P, c)
        stat_dist = np.reshape(stat_dist, (dx, dy))
    else:
        if not silent:
            print(
                f"Convergence reached in {iter_ctr} iterations. No stationary distribution computation requested."
            )
        stat_dist = None
        exp_cost = g[0].item()

    if not silent:
        print(
            f"[entropic_otc] Finished. Total time elapsed: {time.time() - start_time:.3f} seconds."
        )

    return exp_cost, P, stat_dist

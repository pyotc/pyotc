"""Main entry point for otc funcitonality"""

from .otc_backend.policy_iteration.dense.exact import exact_otc as _exact_otc_dense
from .otc_backend.policy_iteration.sparse.exact import exact_otc as _exact_otc_sparse
from .otc_backend.policy_iteration.dense.entropic import entropic_otc as _entropic_otc


def exact_otc(
    P1,
    P2,
    c,
    *,
    stat_dist="best",
    backend="dense",
    max_iter=None,
):
    """
    Computes the optimal transport coupling (OTC) between two stationary Markov chains represented by transition matrices Px and Py,
    as described in Algorithm 1 of the paper: "Optimal Transport for Stationary Markov Chains via Policy Iteration"
    (https://www.jmlr.org/papers/volume23/21-0519/21-0519.pdf).

    The algorithm iteratively updates the transition coupling matrix until convergence by alternating
    between Transition Coupling Evaluation (TCE) and Transition Coupling Improvement (TCI) steps.

    For a detailed discussion of the connection between the OTC problem and Markov Decision Processes (MDPs), see Section 4 of the paper.
    Additional background on policy iteration methods for solving average-cost MDP problems can be found in Chapters 8 and 9 of
    "Markov Decision Processes: Discrete Stochastic Dynamic Programming" by Martin L. Puterman.

    Args:
        P1 (np.ndarray): Transition matrix of the source Markov chain of shape (dx, dx).
        P2 (np.ndarray): Transition matrix of the target Markov chain of shape (dy, dy).
        c (np.ndarray): Cost function of shape (dx, dy).
        stat_dist (str, optional): Method to compute the stationary distribution.
                                   Options include 'best', 'eigen', 'iterative' and None. Defaults to 'best'.
        backend (str, optional): Backend to use for computation. Options are "dense" (default) and "sparse".
        max_iter (int, optional): Maximum number of iterations for the sparse backend. Only applicable when backend is "sparse".

    Returns:
        exp_cost (float): Expected transport cost under the optimal transition coupling.
        R (np.ndarray or scipy.sparse.csr_matrix): Optimal transition coupling matrix of shape (dx*dy, dx*dy).
        stat_dist (np.ndarray): Stationary distribution of the optimal transition coupling of shape (dx, dy).

        Returns (None, None, None) if the algorithm fails to converge.
    """
    if backend == "dense":
        return _exact_otc_dense(P1, P2, c, stat_dist=stat_dist)
    elif backend == "sparse":
        if max_iter is None:
            return _exact_otc_sparse(P1, P2, c, stat_dist=stat_dist)
        return _exact_otc_sparse(P1, P2, c, stat_dist=stat_dist, max_iter=max_iter)
    else:
        raise ValueError("Unknown backend: {backend}. Choose from {'dense', 'sparse'}.")


def entropic_otc(
    Px,
    Py,
    c,
    *,
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
        method (str): Method for the Sinkhorn algorithm. Must choose from ['logsinkhorn', 'ot_sinkhorn', 'ot_logsinkhorn', 'ot_greenkhorn']. Default is 'logsinkhorn'.
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
    return _entropic_otc(
        Px,
        Py,
        c,
        L=L,
        T=T,
        xi=xi,
        method=method,
        sink_iter=sink_iter,
        reg_num=reg_num,
        get_sd=get_sd,
        silent=silent,
    )

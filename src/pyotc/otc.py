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
    if backend == "dense":
        return _exact_otc_dense(P1, P2, c, stat_dist=stat_dist)
    elif backend == "sparse":
        if max_iter is None:
            return _exact_otc_sparse(P1, P2, c, stat_dist=stat_dist)
        return _exact_otc_sparse(
            P1, P2, c, stat_dist=stat_dist, max_iter=max_iter
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


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

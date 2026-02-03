"""Test and benchmark exact OTC technique"""

import numpy as np
import networkx as nx
import time
import pytest

from pyotc.otc_backend.policy_iteration.dense.exact import exact_otc_lp

# from pyotc.otc_backend.policy_iteration.dense.exact import exact_otc as exact_otc_dense
# from pyotc.otc_backend.policy_iteration.sparse.exact import (
#     exact_otc as exact_otc_sparse,
# )
from pyotc import exact_otc
from pyotc.otc_backend.graph.utils import adj_to_trans, get_degree_cost
from pyotc.examples.stochastic_block_model import stochastic_block_model
from pyotc.examples.wheel import wheel_1, wheel_2, wheel_3
from pyotc.examples.edge_awareness import graph_1, graph_2, graph_3, c21, c23

# 1. Test exact OTC on stochastic block model
np.random.seed(1009)
prob_mat = np.array(
    [
        [0.9, 0.1, 0.1, 0.1],
        [0.1, 0.9, 0.1, 0.1],
        [0.1, 0.1, 0.9, 0.1],
        [0.1, 0.1, 0.1, 0.9],
    ]
)
M = [2, 4, 8, 16]
sbms = [
    {
        "A1": stochastic_block_model(sizes=(m, m, m, m), probs=prob_mat),
        "A2": stochastic_block_model(sizes=(m, m, m, m), probs=prob_mat),
    }
    for m in M
]
trans = [{"P1": adj_to_trans(s["A1"]), "P2": adj_to_trans(s["A2"])} for s in sbms]
costs = [get_degree_cost(s["A1"], s["A2"]) for s in sbms]

test_data = zip(trans, costs)


@pytest.mark.parametrize("transition, cost", test_data)
def test_sbm_exact_otc(transition, cost):
    # python optimal transport algo (numpy)
    start = time.time()
    # exp_cost1, _, _ = exact_otc_dense(transition["P1"], transition["P2"], cost)
    exp_cost1, _, _ = exact_otc(transition["P1"], transition["P2"], cost)
    end = time.time()
    print(f"`exact_otc` (pot) run time: {end - start}")

    # python optimal transport algo (scipy.sparse)
    start = time.time()
    # exp_cost2, _, _ = exact_otc_sparse(transition["P1"], transition["P2"], cost)
    exp_cost2, _, _ = exact_otc(
        transition["P1"], transition["P2"], cost, backend="sparse", max_iter=100
    )
    end = time.time()
    print(f"`exact_otc` (pot) run time: {end - start}")

    # check consistency
    assert np.allclose(exp_cost1, exp_cost2)


# 2. Test exact OTC on wheel graph
wheel_A = [
    nx.to_numpy_array(wheel_1),
    nx.to_numpy_array(wheel_2),
    nx.to_numpy_array(wheel_3),
]
wheel_P = [adj_to_trans(A) for A in wheel_A]
wheel_c = [
    get_degree_cost(wheel_A[0], wheel_A[1]),
    get_degree_cost(wheel_A[0], wheel_A[2]),
]


def test_wheel_exact_otc():
    # python optimal transport algo
    exp_cost12_dense, _, _ = exact_otc(
        wheel_P[0], wheel_P[1], wheel_c[0], stat_dist=None, backend="dense"
    )
    exp_cost12_sparse, _, _ = exact_otc(
        wheel_P[0], wheel_P[1], wheel_c[0], stat_dist=None, backend="sparse"
    )

    exp_cost13_dense, _, _ = exact_otc(
        wheel_P[0], wheel_P[2], wheel_c[1], stat_dist=None, backend="dense"
    )
    exp_cost13_sparse, _, _ = exact_otc(
        wheel_P[0], wheel_P[2], wheel_c[1], stat_dist=None, backend="sparse"
    )

    # check consistency
    assert np.allclose(exp_cost12_dense, 2.6551724137931036)
    assert np.allclose(exp_cost12_dense, exp_cost12_sparse)
    assert np.allclose(exp_cost13_dense, 2.5517241379310316)
    assert np.allclose(exp_cost13_dense, exp_cost13_sparse)


# 3. Test exact OTC on edge awareness example
edge_awareness_A = [
    nx.to_numpy_array(graph_1),
    nx.to_numpy_array(graph_2),
    nx.to_numpy_array(graph_3),
]
edge_awareness_P = [adj_to_trans(A) for A in edge_awareness_A]
edge_awareness_c = [c21, c23]


def test_edge_awareness_exact_otc():
    # python optimal transport algo
    exp_cost21, _, _ = exact_otc(
        edge_awareness_P[1],
        edge_awareness_P[0],
        edge_awareness_c[0],
        stat_dist="eigen",
        backend="dense",
    )
    exp_cost21_lp, _, _ = exact_otc_lp(
        edge_awareness_P[1], edge_awareness_P[0], edge_awareness_c[0], stat_dist="best"
    )
    exp_cost23, _, _ = exact_otc(
        edge_awareness_P[1],
        edge_awareness_P[2],
        edge_awareness_c[1],
        stat_dist="iterative",
        backend="dense",
    )
    exp_cost23_lp, _, _ = exact_otc_lp(
        edge_awareness_P[1], edge_awareness_P[2], edge_awareness_c[1], stat_dist=None
    )

    # check consistency
    assert np.allclose(exp_cost21, 0.5714285714285724)
    assert np.allclose(exp_cost21, exp_cost21_lp)
    assert np.allclose(exp_cost23, 0.4464098659648501)
    assert np.allclose(exp_cost23, exp_cost23_lp)


def test_exact_otc_invalid_backend():
    with pytest.raises(ValueError):
        exact_otc(
            edge_awareness_P[1],
            edge_awareness_P[2],
            edge_awareness_c[1],
            backend="invalid_backend",
        )

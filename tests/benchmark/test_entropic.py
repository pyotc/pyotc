"""Test and benchmark entropic OTC technique"""

import numpy as np
import networkx as nx
import time
import pytest

from pyotc.otc_backend.policy_iteration.dense.entropic import entropic_otc
from pyotc.otc_backend.graph.utils import adj_to_trans, get_degree_cost
from pyotc.examples.stochastic_block_model import stochastic_block_model
from pyotc.examples.wheel import wheel_1, wheel_2, wheel_3

# 1. Test entropic OTC on stochastic block model
np.random.seed(1009)
prob_mat = np.array(
    [
        [0.9, 0.1, 0.1, 0.1],
        [0.1, 0.9, 0.1, 0.1],
        [0.1, 0.1, 0.9, 0.1],
        [0.1, 0.1, 0.1, 0.9],
    ]
)
M = [4, 8, 16]
sbms = [
    {
        "A1": stochastic_block_model(sizes=(m, m, m, m), probs=prob_mat),
        "A2": stochastic_block_model(sizes=(m, m, m, m), probs=prob_mat),
    }
    for m in M
]
trans = [{"P1": adj_to_trans(s["A1"]), "P2": adj_to_trans(s["A2"])} for s in sbms]
costs = [get_degree_cost(s["A1"], s["A2"]) for s in sbms]
entropic_costs = [0.825366655599364, 0.947547841440221, 3.054455073362832]

test_data = zip(trans, costs, entropic_costs)


@pytest.mark.parametrize("transition, cost, entropic_cost", test_data)
def test_sbm_entropic_otc(transition, cost, entropic_cost):
    # entropic otc
    start = time.time()
    exp_cost, _, _ = entropic_otc(
        transition["P1"], transition["P2"], cost, xi=10, sink_iter=100
    )
    end = time.time()
    print(f"`entropic_otc` run time: {end - start}")

    # check consistency
    print(exp_cost)
    assert np.allclose(exp_cost, entropic_cost)


# 2. Test entropic OTC on wheel graph
A1 = nx.to_numpy_array(wheel_1)
A2 = nx.to_numpy_array(wheel_2)
A3 = nx.to_numpy_array(wheel_3)

P1 = adj_to_trans(A1)
P2 = adj_to_trans(A2)
P3 = adj_to_trans(A3)

c12 = get_degree_cost(A1, A2)
c13 = get_degree_cost(A1, A3)


def test_wheel_exact_otc():
    # entropic otc
    start = time.time()
    exp_cost12, _, _ = entropic_otc(
        P1, P2, c12, get_sd=True, L=25, T=50, xi=0.1, sink_iter=10
    )
    exp_cost13, _, _ = entropic_otc(
        P1, P3, c13, get_sd=True, L=25, T=50, xi=0.1, sink_iter=10
    )
    end = time.time()
    print(f"`entropic_otc` run time: {end - start}")

    # check consistency
    print(exp_cost12, exp_cost13)
    assert np.allclose(exp_cost12, 2.65527250306173)
    assert np.allclose(exp_cost13, 2.553874249034749)

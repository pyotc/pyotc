"""Test and benchmark exact OTC technique
"""
import numpy as np
import networkx as nx
import time
import pytest

from pyotc.otc_backend.policy_iteration.exact import exact_otc_lp
from pyotc.otc_backend.policy_iteration.exact import exact_otc_pot
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
M = [2,4,8,16]
sbms = [{"A1": stochastic_block_model(sizes=(m, m, m, m), probs=prob_mat),
        "A2": stochastic_block_model(sizes=(m, m, m, m), probs=prob_mat),}
        for m in M]
trans = [{"P1": adj_to_trans(s["A1"]), "P2": adj_to_trans(s["A2"])} 
        for s in sbms]
costs = [get_degree_cost(s["A1"], s["A2"]) for s in sbms]

test_data = zip(trans, costs)

@pytest.mark.parametrize("transition, cost", test_data)
def test_sbm_exact_otc(transition, cost):
    # scipy linprog algo
    start = time.time()
    exp_cost1, _, _ = exact_otc_lp(transition["P1"], transition["P2"], cost)
    end = time.time()
    print(f"`exact_otc` (scipy) run time: {end - start}")

    # python optimal transport algo
    start = time.time()
    exp_cost2, _, _ = exact_otc_pot(transition["P1"], transition["P2"], cost)
    end = time.time()
    print(f"`exact_otc_pot` (pot) run time: {end - start}")

    # check consistency
    assert np.allclose(exp_cost1, exp_cost2)

    

# 2. Test exact OTC on wheel graph
A1 = nx.to_numpy_array(wheel_1)
A2 = nx.to_numpy_array(wheel_2)
A3 = nx.to_numpy_array(wheel_3)

P1 = adj_to_trans(A1)
P2 = adj_to_trans(A2)
P3 = adj_to_trans(A3)

c12 = get_degree_cost(A1, A2)
c13 = get_degree_cost(A1, A3)

def test_wheel_exact_otc():
    # python optimal transport algo
    start = time.time()
    exp_cost12, _, _ = exact_otc_pot(P1, P2, c12)
    exp_cost13, _, _ = exact_otc_pot(P1, P3, c13)
    end = time.time()
    print(f"`exact_otc_pot` (pot) run time: {end - start}")

    # check consistency
    print(exp_cost12, exp_cost13)
    assert np.allclose(exp_cost12, 2.6551724137931036)
    assert np.allclose(exp_cost13, 2.551724137931033)
    

# 3. Test exact OTC on edge awareness example
A1 = nx.to_numpy_array(graph_1)
A2 = nx.to_numpy_array(graph_2)
A3 = nx.to_numpy_array(graph_3)

P1 = adj_to_trans(A1)
P2 = adj_to_trans(A2)
P3 = adj_to_trans(A3)

def test_edge_awareness_exact_otc():
    # python optimal transport algo
    start = time.time()
    exp_cost21, _, _ = exact_otc_pot(P2, P1, c21)
    exp_cost23, _, _ = exact_otc_pot(P2, P3, c23)
    end = time.time()
    print(f"`exact_otc_pot` (pot) run time: {end - start}")

    # check consistency
    print(exp_cost21, exp_cost23)
    assert np.allclose(exp_cost21, 0.5714285714285714)
    assert np.allclose(exp_cost23, 0.4464098659648351)
    
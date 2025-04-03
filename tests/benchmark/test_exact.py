"""Test and benchmark exact OTC technique"""

import numpy as np
import time
import pytest

from pyotc.otc_backend.policy_iteration.exact import exact_otc_lp
from pyotc.otc_backend.policy_iteration.exact import exact_otc_pot
from pyotc.otc_backend.graph.utils import adj_to_trans, get_degree_cost
from pyotc.examples.stochastic_block_model import stochastic_block_model

np.random.seed(1009)

def two_comm_sbm():
    P = np.array([[.8, .2],[.2, .8]])
    return P

def assortative_sbm():
    """Creates a 4-community assortative stochastic block model matrix."""
    P = np.array([
        [0.8, 0.1, 0.05, 0.05],
        [0.1, 0.8, 0.05, 0.05],
        [0.05, 0.05, 0.8, 0.1],
        [0.05, 0.05, 0.1, 0.8]
    ])
    return P

def hierarchical_sbm():
    """Creates a 4-community hierarchical stochastic block model matrix."""
    P = np.array([
        [0.7, 0.6, 0.2, 0.1],
        [0.6, 0.7, 0.2, 0.1],
        [0.2, 0.2, 0.7, 0.6],
        [0.1, 0.1, 0.6, 0.7]
    ])
    return P
prob_mat_0 = two_comm_sbm()
prob_mat_1 = assortative_sbm()
prob_mat_2 = hierarchical_sbm()

M = [2, 4, 8, 16]
sbms = [
    {
        "A1": stochastic_block_model(sizes=(m, m, m, m), probs=prob_mat_1),
        "A2": stochastic_block_model(sizes=(m, m, m, m), probs=prob_mat_1),
    }
    for m in M
]
trans = [{"P1": adj_to_trans(s["A1"]), "P2": adj_to_trans(s["A2"])} for s in sbms]
costs = [get_degree_cost(a["P1"], a["P2"]) for a in trans]
expected_costs = [0.0, 0.0, 0.0, 0.0]
test_data = zip(M, trans, costs, expected_costs)
marked_test_data = [data if data[0] <= 4 else pytest.param(*data, marks=pytest.mark.slow) for data in test_data]


@pytest.mark.parametrize("m, transition, cost, expected_cost", marked_test_data)
def test_exact_otc_lp(m, transition, cost, expected_cost):
    exp_cost1, otc1, stat_dist1 = exact_otc_lp(transition["P1"], transition["P2"], cost)
    assert np.allclose(exp_cost1, expected_cost)


@pytest.mark.parametrize("m, transition, cost, expected_cost", marked_test_data)
def test_exact_otc_pot(m, transition, cost, expected_cost):
    exp_cost1, otc1, stat_dist1 = exact_otc_pot(transition["P1"], transition["P2"], cost)
    assert np.allclose(exp_cost1, expected_cost)


# @pytest.mark.parametrize("m, sbm, transition, cost", test_data)
# def test_time_exact_otc(m, sbm, transition, cost):
#     # scipy linprog algo
#     start = time.time()
#     exp_cost1, otc1, stat_dist1 = exact_otc_lp(transition["P1"], transition["P2"], cost)
#     end = time.time()
#     print(f"`exact_otc` (scipy) run time: {end - start}")

#     # python optimal transport algo
#     start = time.time()
#     exp_cost2, otc2, stat_dist2 = exact_otc_pot(
#         transition["P1"], transition["P2"], cost
#     )
#     end = time.time()
#     print(f"`exact_otc_pot` (pot) run time: {end - start}")

#     start = time.time()
#     exp_cost3, otc3, stat_dist3 = exact_otc_pot(
#         transition["P1"], transition["P2"], cost, get_best_sd=True
#     )
#     end = time.time()
#     print(f"`exact_otc_pot` (pot, get_best_sd) run time: {end - start}")

#     # check consistency
#     assert np.allclose(exp_cost1, exp_cost2)
#     assert np.allclose(exp_cost2, exp_cost3)

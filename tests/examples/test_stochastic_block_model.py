"""Test stochastic block model"""

from pyotc.examples.stochastic_block_model import stochastic_block_model
import numpy as np
import networkx as nx

# Seed number
np.random.seed(1009)

m = 10
A1 = stochastic_block_model(
    (m, m, m, m),
    np.array(
        [
            [0.9, 0.1, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1],
            [0.1, 0.1, 0.9, 0.1],
            [0.1, 0.1, 0.1, 0.9],
        ]
    ),
)

sbm_1 = nx.from_numpy_array(A1)


def test_shape_A1():
    assert A1.shape == (40, 40)


def test_A1_symmetry():
    assert np.array_equal(A1, A1.T)


def test_A1_graph():
    assert len(sbm_1.edges()) == 216

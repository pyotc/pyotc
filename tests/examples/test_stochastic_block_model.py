"""Test stochastic block model"""

import pytest

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


def test_no_self_loops():
    assert np.all(np.diag(A1) == 0)


def test_entries_binary():
    unique_vals = np.unique(A1)
    assert set(unique_vals.tolist()).issubset({0, 1})


def test_invalid_probs_type_and_nonsquare():
    # Not a numpy array
    with pytest.raises(ValueError, match="'probs' must be a square numpy array."):
        stochastic_block_model((2, 2), [[0.9, 0.1], [0.1, 0.9]])
    # Nonsquare numpy array
    with pytest.raises(ValueError, match="'probs' must be a square numpy array."):
        stochastic_block_model((2, 2), np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]]))


def test_invalid_probs_symmetry():
    probs = np.array([[0.9, 0.2], [0.1, 0.9]])  # not symmetric
    with pytest.raises(ValueError, match="'probs' must be a symmetric matrix."):
        stochastic_block_model((1, 1), probs)


def test_sizes_probs_dimension_mismatch():
    probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    with pytest.raises(
        ValueError, match="'sizes' and 'probs' dimensions do not match."
    ):
        stochastic_block_model((1,), probs)

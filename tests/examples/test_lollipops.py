from pyotc.examples.lollipops import lollipop_1, lollipop_2
import pytest
import networkx as nx
from pyotc.examples.lollipops import lollipop_1, lollipop_2, left_lollipop_graph, right_lollipop_graph
class TestLollipopGraphs:
    """Test lollipop graph definitions and properties."""

    def test_lollipop_1_node_count(self):
        """Test that lollipop_1 has 12 nodes."""
        assert lollipop_1.number_of_nodes() == 12

    def test_lollipop_2_node_count(self):
        """Test that lollipop_2 has 12 nodes."""
        assert lollipop_2.number_of_nodes() == 12

    def test_lollipop_1_edge_count(self):
        """Test that lollipop_1 has 12 edges."""
        assert lollipop_1.number_of_edges() == 12

    def test_lollipop_2_edge_count(self):
        """Test that lollipop_2 has 12 edges."""
        assert lollipop_2.number_of_edges() == 12

    def test_lollipop_1_nodes(self):
        """Test that lollipop_1 contains nodes 1-12."""
        expected_nodes = set(range(1, 13))
        assert set(lollipop_1.nodes()) == expected_nodes

    def test_lollipop_2_nodes(self):
        """Test that lollipop_2 contains nodes 1-12."""
        expected_nodes = set(range(1, 13))
        assert set(lollipop_2.nodes()) == expected_nodes

    def test_left_lollipop_graph_structure(self):
        """Test left_lollipop_graph dictionary structure."""
        assert "nodes" in left_lollipop_graph
        assert "edges" in left_lollipop_graph
        assert "name" in left_lollipop_graph
        assert len(left_lollipop_graph["nodes"]) == 12
        assert len(left_lollipop_graph["edges"]) == 12

    def test_right_lollipop_graph_structure(self):
        """Test right_lollipop_graph dictionary structure."""
        assert "nodes" in right_lollipop_graph
        assert "edges" in right_lollipop_graph
        assert "name" in right_lollipop_graph
        assert len(right_lollipop_graph["nodes"]) == 12
        assert len(right_lollipop_graph["edges"]) == 12

    def test_lollipop_1_specific_edges(self):
        """Test that lollipop_1 contains specific edges from data."""
        assert lollipop_1.has_edge(1, 2)
        assert lollipop_1.has_edge(12, 4)
        assert lollipop_1.has_edge(11, 12)

    def test_lollipop_2_specific_edges(self):
        """Test that lollipop_2 contains specific edges from data."""
        assert lollipop_2.has_edge(7, 9)
        assert lollipop_2.has_edge(12, 6)
        assert lollipop_2.has_edge(5, 3)

"""
networkx graphs for lollipop examples given in

[Alignment and Comparison of Directed Networks via Transition Couplings of Random Walks](https://arxiv.org/abs/2106.07106)

Figure 5 graphs

* lollipop_1 is the left graph
* lollipop_2 is the right graph
"""

import networkx as nx

# Define graphs
left_lollipop_graph = {
    "nodes": [{"id": i} for i in range(1, 13)],
    "edges": [
        {"source": 1, "target": 2},
        {"source": 2, "target": 3},
        {"source": 3, "target": 4},
        {"source": 4, "target": 5},
        {"source": 5, "target": 6},
        {"source": 6, "target": 7},
        {"source": 7, "target": 8},
        {"source": 8, "target": 9},
        {"source": 9, "target": 10},
        {"source": 10, "target": 11},
        {"source": 11, "target": 12},
        {"source": 12, "target": 4},
    ],
    "name": "left lollipop graph",
}

right_lollipop_graph = {
    "nodes": [{"id": i} for i in range(1, 13)],
    "edges": [
        {"source": 7, "target": 9},
        {"source": 9, "target": 4},
        {"source": 4, "target": 6},
        {"source": 6, "target": 1},
        {"source": 1, "target": 2},
        {"source": 2, "target": 11},
        {"source": 11, "target": 8},
        {"source": 8, "target": 10},
        {"source": 10, "target": 5},
        {"source": 5, "target": 3},
        {"source": 3, "target": 12},
        {"source": 12, "target": 6},
    ],
    "name": "right lollipop graph",
}

lollipop_1 = nx.node_link_graph(data=left_lollipop_graph, edges="edges")
lollipop_2 = nx.node_link_graph(data=right_lollipop_graph, edges="edges")

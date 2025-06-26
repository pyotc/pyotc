"""
networkx graphs for wheel graph examples given in Version 1 of

[Alignment and Comparison of Directed Networks via Transition Couplings of Random Walks](https://arxiv.org/pdf/2106.07106v1.pdf)

Figure 2 graphs

* G_1 is a wheel graph of order 16
* G_2 is a wheel graph removing 1 spoke edge
* G_3 is a wheel graph removing 1 wheel edge
"""

import networkx as nx

# Define graphs G1, G2, G3
wheel_graph_1 = {
    "nodes": [{"id": i} for i in range(1, 17)],
    "edges": [
        {"source": 1, "target": 2},
        {"source": 1, "target": 3},
        {"source": 1, "target": 4},
        {"source": 1, "target": 5},
        {"source": 1, "target": 6},
        {"source": 1, "target": 7},
        {"source": 1, "target": 8},
        {"source": 1, "target": 9},
        {"source": 1, "target": 10},
        {"source": 1, "target": 11},
        {"source": 1, "target": 12},
        {"source": 1, "target": 13},
        {"source": 1, "target": 14},
        {"source": 1, "target": 15},
        {"source": 1, "target": 16},
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
        {"source": 12, "target": 13},
        {"source": 13, "target": 14},
        {"source": 14, "target": 15},
        {"source": 15, "target": 16},
        {"source": 16, "target": 2},
    ],
    "name": "wheel graph 1",
}

wheel_graph_2 = {
    "nodes": [{"id": i} for i in range(1, 17)],
    "edges": [
        {"source": 1, "target": 3},
        {"source": 1, "target": 4},
        {"source": 1, "target": 5},
        {"source": 1, "target": 6},
        {"source": 1, "target": 7},
        {"source": 1, "target": 8},
        {"source": 1, "target": 9},
        {"source": 1, "target": 10},
        {"source": 1, "target": 11},
        {"source": 1, "target": 12},
        {"source": 1, "target": 13},
        {"source": 1, "target": 14},
        {"source": 1, "target": 15},
        {"source": 1, "target": 16},
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
        {"source": 12, "target": 13},
        {"source": 13, "target": 14},
        {"source": 14, "target": 15},
        {"source": 15, "target": 16},
        {"source": 16, "target": 2},
    ],
    "name": "wheel graph 2",
}

wheel_graph_3 = {
    "nodes": [{"id": i} for i in range(1, 17)],
    "edges": [
        {"source": 1, "target": 2},
        {"source": 1, "target": 3},
        {"source": 1, "target": 4},
        {"source": 1, "target": 5},
        {"source": 1, "target": 6},
        {"source": 1, "target": 7},
        {"source": 1, "target": 8},
        {"source": 1, "target": 9},
        {"source": 1, "target": 10},
        {"source": 1, "target": 11},
        {"source": 1, "target": 12},
        {"source": 1, "target": 13},
        {"source": 1, "target": 14},
        {"source": 1, "target": 15},
        {"source": 1, "target": 16},
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
        {"source": 12, "target": 13},
        {"source": 13, "target": 14},
        {"source": 14, "target": 15},
        {"source": 15, "target": 16},
    ],
    "name": "wheel graph 3",
}

wheel_1 = nx.node_link_graph(data=wheel_graph_1, edges="edges")
wheel_2 = nx.node_link_graph(data=wheel_graph_2, edges="edges")
wheel_3 = nx.node_link_graph(data=wheel_graph_3, edges="edges")

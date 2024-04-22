"""
networkx graphs for edge awareness example and corresponding costs from Table 1

[Alignment and Comparison of Directed Networks via Transition Couplings of Random Walks](https://arxiv.org/abs/2106.07106)

Figure 4 graphs

* G_1 is the regular octogon
* G_2 is the regular octogon removing 1 edge
* G_3 is uniform edge lengths of the octogon removing 1 edge
"""

import numpy as np
import networkx as nx

# Define graphs G1, G2, G3
edge_awareness_1 = {
    "nodes": [
        {"id": 1},
        {"id": 2},
        {"id": 3},
        {"id": 4},
        {"id": 5},
        {"id": 6},
        {"id": 7},
        {"id": 8}
    ],
    "links": [
        {"source": 1, "target": 2},
        {"source": 2, "target": 3},
        {"source": 3, "target": 4},
        {"source": 4, "target": 5},
        {"source": 5, "target": 6},
        {"source": 6, "target": 7},
        {"source": 7, "target": 8},
        {"source": 8, "target": 1},
    ],
    "name": "edge awareness graph 1",
}

edge_awareness_2_3 = {
    "nodes": [
        {"id": 1},
        {"id": 2},
        {"id": 3},
        {"id": 4},
        {"id": 5},
        {"id": 6},
        {"id": 7},
        {"id": 8}
    ],
    "links": [
        {"source": 1, "target": 2},
        {"source": 2, "target": 3},
        {"source": 3, "target": 4},
        {"source": 4, "target": 5},
        {"source": 5, "target": 6},
        {"source": 6, "target": 7},
        {"source": 7, "target": 8}
    ],
    "name": "edge awareness graph 2, 3",
}

graph_1 = nx.node_link_graph(data=edge_awareness_1)
graph_2 = nx.node_link_graph(data=edge_awareness_2_3)
graph_3 = nx.node_link_graph(data=edge_awareness_2_3)

# Define the coordinates
d1 = np.zeros((8, 2))
for i in range(8):
    d1[i, 0] = np.cos(np.pi / 8 + np.pi / 4 * i)
    d1[i, 1] = np.sin(np.pi / 8 + np.pi / 4 * i)

d2 = d1.copy()

d3 = np.zeros((8, 2))
for i in range(8):
    d3[i, 0] = np.cos(np.pi / 2 + np.pi / 7 * i)
    d3[i, 1] = np.sin(np.pi / 2 + np.pi / 7 * i)

# Get cost matrices
n = 8
c1 = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        c1[i, j] = np.sum((d2[i, :] - d1[j, :]) ** 2)

c2 = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        c2[i, j] = np.sum((d2[i, :] - d3[j, :]) ** 2)

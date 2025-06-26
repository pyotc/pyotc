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
    "nodes": [{"id": i} for i in range(1, 9)],
    "edges": [
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
    "nodes": [{"id": i} for i in range(1, 9)],
    "edges": [
        {"source": 1, "target": 2},
        {"source": 2, "target": 3},
        {"source": 3, "target": 4},
        {"source": 4, "target": 5},
        {"source": 5, "target": 6},
        {"source": 6, "target": 7},
        {"source": 7, "target": 8},
    ],
    "name": "edge awareness graph 2, 3",
}

graph_1 = nx.node_link_graph(data=edge_awareness_1, edges="edges")
graph_2 = nx.node_link_graph(data=edge_awareness_2_3, edges="edges")
graph_3 = nx.node_link_graph(data=edge_awareness_2_3, edges="edges")

# Define the coordinates of G_1, G_2, G_3
# All vertices are located on the unit circle in R^2
# d1: coordinate of G_1 vertices (regular octagon)
# d2: coordinate of G_2 vertices (regular octagon)
# d3: coordinate of G_3 vertices (the vertices are uniformly distributed in the left semicircle)

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
# Define a cost function equal to the squared Euclidean distance between vertex positions
# c21: cost function between G_2 and G_1
# c23: cost function between G_2 and G_3


def euclidean_cost(v1, v2):
    n1 = v1.shape[0]
    n2 = v2.shape[0]
    c = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            c[i, j] = np.sum((v1[i, :] - v2[j, :]) ** 2)

    return c


c21 = euclidean_cost(d2, d1)
c23 = euclidean_cost(d2, d3)

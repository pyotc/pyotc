---
title: 'PyOTC: A Python Package for Optimal Transition Coupling'
tags:
  - Python
  - Optimal Transport
  - Probability
authors:
  - name: Bongsoo Yi
    affiliation: 1
  - name: Yuning Pan
    affiliation: 2
  - name: Jay Hineman
    affiliation: 3
affiliations:
 - name: Department of Statistics and Operations Research, University of North Carolina at Chapel Hill, Chapel Hill, NC, USA
   index: 1
 - name: Department of Mathematics and Statistics, Boston University, Boston, MA, USA
   index: 2
 - name: Applied Research Associates, Raleigh, NC, USA
   index: 3
date: 25 May 2026
bibliography: paper.bib
---

# Summary
Recent scholarly work [@oconnor_optimal_2022] introduced an extension of optimal transport that applies directly to stationary Markov processes. This extension enables the computation of meaningful distances between such processes, facilitating comparisons of networks and graphs in fields such as chemistry, biology, and social science. We provide a performant Python implementation of this method [@oconnor_optimal_2022], along with interfaces for related network and graph problems [@yi_alignment_2024]. Our implementation is open source, tested, and integrates with the Python data science ecosystem.

# State of the Field

Optimal transport (OT) has become a widely used computational framework across machine learning, statistics, and network science. A mature ecosystem of OT software already exists, most notably `POT` (Python Optimal Transport) [@flamary_pot_2021], which provides efficient solvers for classical optimal transport problems and their entropic approximations. However, optimal transport between *Markov processes*—specifically optimal transition coupling (OTC) introduced by @oconnor_optimal_2022—requires additional structure beyond standard OT formulations.

Two implementations of OTC currently exist in MATLAB [@oconnor_oconnor-kevinotc_2022; @yi_austinyinetotc_2023]. While these implementations supported the original research developments, they share limitations typical of the MATLAB ecosystem: they are not always freely accessible and lack integration with the broader Python-based scientific computing ecosystem. This makes it difficult to incorporate OTC methods into modern data science workflows.

`pyotc` addresses this gap by providing the first open-source Python implementation of optimal transition coupling. The package builds on existing optimal transport infrastructure such as `POT` [@flamary_pot_2021], while implementing the additional algorithmic structure required for OTC. This allows researchers to apply OTC methods within the broader Python scientific computing ecosystem.

# Software Design

The design of `pyotc` prioritizes modularity, interoperability with existing optimal transport infrastructure, and scalability to larger graph-based problems. A key design decision was to build on top of the `POT` library [@flamary_pot_2021] rather than implementing low-level optimal transport solvers from scratch. Leveraging `POT` allows `pyotc` to reuse highly optimized algorithms such as network simplex and Sinkhorn iterations while focusing development effort on the algorithmic structure specific to optimal transition coupling.

The architecture closely mirrors the theoretical algorithm introduced in @oconnor_optimal_2022. The OTC problem is solved using a policy iteration procedure consisting of alternating evaluation and improvement steps. This separation makes the implementation easy to understand and extend while allowing the improvement stage to reuse existing optimal transport solvers.

`pyotc` supports both exact and entropically regularized solution methods. The exact approach is useful for validating implementations and enabling further algorithmic development, while entropic regularization provides a scalable approximation that can handle larger problems more efficiently. The choice of the entropic regularization parameter affects convergence behavior and remains an active research topic in approximate optimal transport and related formulations such as the Schrödinger Bridge problem [@peyre_computational_2020; @nutz_introduction_2021].

To support practical applications in network analysis, the implementation also provides optional sparse matrix representations that reduce memory usage and allow larger transition graphs to be handled efficiently. Integration with Python libraries such as `NetworkX` [@hagberg_exploring_2008] enables workflows involving graph construction, analysis, and comparison. Future applications may also involve integration with machine learning libraries such as `scikit-learn` [@pedregosa_scikit-learn_2011].

## Performance Comparison

To demonstrate the performance improvements of `pyotc` over existing MATLAB implementations, we benchmark our Python implementation against the original MATLAB code from [@oconnor_oconnor-kevinotc_2022]. The benchmark evaluates the execution time on two randomly generated stochastic block models (SBMs), each consisting of 4 blocks with an intra-block connection probability of 0.9 and an inter-block connection probability of 0.1. Results show that `pyotc` achieves substantial speedups while yielding identical results to the MATLAB implementation. All benchmarks were executed on a machine equipped with a 2.3 GHz quad-core Intel Core i7 processor and 16 GB of RAM. The MATLAB benchmarks were executed using MATLAB R2025b, with the JIT compiler pre-warmed prior to measurement to ensure accurate timing results.

|    Number of Nodes   |  20  |   32  |   40  |   60   |    80   |   100    |
|:--------:|:----:|:-----:|:-----:|:------:|:-------:|:-------|
| MATLAB (s) | 5.51 | 26.88 | 61.27 | 308.20 | 1246.13 | 3890.41  |
|    PyOTC (s)   | 0.60 |  3.21 |  9.60 |  54.73 |  322.44 | 1340.12 |

# Statement of Need

Optimal transport has proven to be a valuable tool in data science and machine learning. One natural extension beyond probability distributions is to stochastic processes, particularly stationary finite-state Markov chains.

Recent work [@oconnor_optimal_2022] developed theory and algorithms for computing optimal transition couplings between such processes. Transition couplings provide a constrained family of transport plans that match marginal distributions while preserving transition dynamics.

These methods have demonstrated practical value in applications such as network alignment and graph comparison [@yi_alignment_2024; @hoang_optimal_2025]. However, practical access to OTC algorithms remains limited. Our goal is to provide a practical, open-source Python implementation that allows researchers to apply these methods in real-world data science workflows. The package is designed to integrate naturally with Python-based scientific computing tools and to support experimentation with OTC algorithms in a wide range of network analysis problems.

## Comparison with other available codes and lineage
The original MATLAB code for OTC was written by Kevin O'Connor and used for the work in [@oconnor_optimal_2022]; the code is available on GitHub [@oconnor_oconnor-kevinotc_2022]. This MATLAB implementation was later extended and applied to network alignment in [@yi_alignment_2024]. The development of `pyotc` was initiated from these predecessors.

Additional related codes have been developed, primarily focusing on the entropic case. Notably, @calo_bisimulation_2024 provide a variation of the Sinkhorn iteration described in [@oconnor_optimal_2022], with an accompanying implementation available on GitHub [@calogithub]. Another related effort is the differentiable extension of entropic OTC described in [@brugere_distances_2024] and implemented in [@brugere_github]. In contrast to these entropic variants, `pyotc` addresses the exact case with a `POT`-powered implementation. Relying on this exact formulation, rather than entropic approximations, is essential for maintaining strict numerical precision in downstream tasks.

# Research Impact Statement

The `pyotc` package serves a broad audience of researchers working with network-structured data across various domains—including computational chemistry, structural biology, neuroimaging, and social sciences-where comparing, aligning, and measuring distances between graphs or Markov chains is a fundamental task. While the core optimal transition coupling (OTC) methodology was initially introduced in [@oconnor_optimal_2022] and applied to network alignment [@yi_alignment_2024], `pyotc` lowers the barrier to entry by providing a robust, Python-native framework that integrates seamlessly with modern scientific pipelines.

The package has already demonstrated direct research impact, bridging the gap between theoretical frameworks and empirical applications. The structural properties and algorithms provided in `pyotc` have been directly utilized, extended, or cited by several recent studies in graph matching, Markov chain analysis, and network alignment [e.g., [@hoang_optimal_2025; @xiang_graph_2026; @calo_bisimulation_2024; @brugere_distances_2024; @parrot_2025]]. 

Beyond literature-based impact, `pyotc` has garnered significant interest from the broader scientific community. The repository has received explicit inquiries and requests for code sharing from external research groups, particularly those specializing in network science and neuroimaging analysis, who seek to apply OTC to empirical brain connectivity data. By transitioning the ecosystem from legacy, closed-loop MATLAB implementations to an open-source, `POT`-backed Python library, `pyotc` enables domain scientists to easily plug network datasets into modern workflows (such as `NetworkX` or `scikit-learn`), facilitating interdisciplinary applications and accelerating the adoption of optimal transport methods in practical network analysis.

# Features

Our implementation includes the tools needed to reproduce the examples from [@oconnor_optimal_2022] and [@yi_alignment_2024] in Python. It achieves strong performance by leveraging optimized optimal transport backends such as the network simplex implementation in `POT` [@flamary_pot_2021].

The `pyotc` code provides two approaches to solving the OTC problem. The *exact* solution procedure solves the underlying optimal transport problems via linear programming. The specialized *network simplex* algorithm from `POT` is used in the improvement step of policy iteration. In the evaluation step, a block linear system involving the transition matrix $R$ and the cost vector $c$ is solved. Alternatively, `pyotc` also provides an iterative method based on entropic regularization. This approach allows the method to scale to larger problems while leveraging the extensive optimal transport functionality available in `POT`.

Algorithm 1 summarizes the exact OTC solution procedure introduced by @oconnor_optimal_2022.

## Algorithm 1: Exact OTC

1. Initialize $R_0 = P \otimes Q$, Convergence tolerance $\tau$
2. Set `converged = False`, $R=R_0$
3. **While** not `converged`:
    1. **Evaluate transition coupling**: $g, h = \text{evaluate}(R)$
    2. **Improve transition coupling**: $R_0 = R$, $R = \text{improve}(g, h)$
    3. **Check convergence**: $d = \|R - R_0\|$, $\text{converged} = d < \tau$
4. **Output**: $R$ (an optimal transition coupling)

# Examples
We provide a basic hello world example here. Our implementation is well documented and simple, consisting essentially of Python functions, which makes it easy to modify.
```python
from pyotc import exact_otc
import numpy as np

P = np.array([[.5, .5], [.5, .5]])
Q = np.array([[0, 1], [1, 0]])
c = np.array([[1, 0], [0, 1]])

exp_cost, R, stat_dist = exact_otc(P, Q, c)
print("\nExact OTC cost between P and Q:", exp_cost)
```

# Conclusion
`pyotc` provides a performant Python implementation for computing optimal transition couplings for stationary Markov chains and their associated graph structures. Optimal transition coupling is a classic example of opportunity to bring a novel computational tool to wider audience through open source software and improve it. By moving to an open ecosystem such as Python, we have produced an OTC code that is faster and, arguably, more capable than existing implementations.

As OTC is an active research topic, we believe there are significant opportunities to extend the work here. In this direction, we hope that this code will facilitate further explorations in both novel algorithms and more general implementations. One could explore for example variations on the policy improvement and policy evaluation algorithms in terms of the stationary distribution (essentially a resolvent calculation). Implementation-wise, there are significant opportunities to provide additional interfaces to Python ecosystem, for example interfaces chem or bio informatics sources (for example RDKit [@landrum_rdkitrdkit_2025]) `pyotc` also enables additional benchmarking studies.


# AI Usage Disclosure

No AI-generated code was used in the development of the software. GPT 5.2 was used to help refine and edit the manuscript.

# References

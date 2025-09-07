---
title: 'PyOTC: A Python Package for Optimal Transition Coupling'
tags:
  - Python
  - Optimal Transport
  - Probabilty
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
 - name: Boston University, Boston, MA, USA
   index: 2
 - name: Applied Research Associates, Raleigh, NC, USA
   index: 3
date: 8 Aug 2025
bibliography: paper.bib
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
---

# Summary
Recent scholarly work [@oconnor_optimal_2022] has introduced an extension of optimal transport that applies directly to stationary Markov processes. This extension enables the computation of meaningful distances between such processes, facilitating comparisons of networks and graphs in various fields like chemistry, biology, and social science. We provide a performant Python implementation of this method [@oconnor_optimal_2022], along with interfaces for related network and graph problems [@yi_alignment_2024]. Our implementation is open source, tested, and integrates with the Python data science ecosystem.

# Statement of need
Optimal transport has proven to be a valuable, practical, and natural tool in data science and machine learning. As a problem in the calculus of variations, conventional optimal transport admits many possible generalizations. One natural extension beyond probability distributions is to processes, particularly pairs of stationary finite-state Markov chains.

Recent work [@oconnor_optimal_2022] has developed initial theory and algorithms for this setting, with a focus on computing optimal transition couplings. Transition couplings form a constrained family of transport plans that not only match the marginal distributions but also capture the dynamics of the Markov chains. Their practical importance has grown further through applications in network alignment and comparison [@yi_alignment_2024], which demonstrate how these couplings can serve as a principled basis for comparing complex directed networks. 

We aim to provide a practical, open-source Python implementation of these methods. Our tool is designed for community extension and includes careful performance baselines. We anticipate this tool will be widely applied in future work on networks in fields like chemistry, neuroscience, and and biology.

`pyotc` addresses several technical needs: providing a Python implementation, accelerating computation, and increasing the maximum problem size that can be handled in memory. Two other implementations for optimal transition coupling exist in MATLAB, which have served as inspiration. However, these share common limitations of MATLAB: they are not open or free (though free alternatives exist) and lack a comprehensive ecosystem for data science.

Python is the de facto language for data science, but our choice of Python is motivated not only by its popularity and rich ecosystem, but also by the availability of highly efficient optimal transport solvers such as `POT` (Python Optimal Transport) [@flamary_pot_2021]. Leveraging `POT` allows `pyotc` to build on a robust and optimized backend for core OT computations. We also integrate with network/graph theory tools such as `networkx` [@hagberg_exploring_2008], and future applications could involve integration with standard machine learning libraries such as `scikit-learn` [@pedregosa_scikit-learn_2011].

To evaluate computational performance, we compare our implementation using different computation and storage configurations, including against the existing MATLAB codes where possible. The results of these comparisons are summarized in Table `\ref`.

We emphasize that the `pyotc` code provides options for both exact solutions and entropic approximations. The exact procedure is important for several reasons: (i) validating the implementation by checking consistency between exact and approximate algorithms, (ii) enabling further algorithmic development, and (iii) benefiting from our optimized Python implementation, which allows exact computations to run efficiently in practice. Existing consistency results are based on stability estimates, but no rate of convergence results are currently available. In practice, the entropic regularization hyperparameter changes convergence behavior and can be tuned accordingly. The selection of regularization is an active area of research in approximate optimal transport and, as a special case, in the Schrödinger Bridge problem [@peyre_computational_2021] [@nutz_introduction_2022].

## Comparison with other available codes and lineage
The original MATLAB code for OTC was written by Kevin O'Connor and used for the work in [@oconnor_optimal_2022]; the code is available on GitHub [@connor_oconnor-kevinotc_2022]. This MATLAB implementation was later extended and applied to network alignment in [@yi_alignment_2024]. The development of `pyotc` was initiated from these predecessors.

Additional related codes have been developed, primarily focusing on the entropic case. Notably, Calo et al. [@calo_bisimulation_2024] provide a variation of the Sinkhorn iteration described in [@oconnor_optimal_2022], with an accompanying implementation available on GitHub [@calogithub]. Another related effort is the differentiable extension of entropic OTC described in [@brugere_distances_2024] and implemented in [@brugere_github].

# Features
Our implementation includes the tools needed to reproduce the examples from [@oconnor_optimal_2022] and [@yi_alignment_2024] in Python. It achieves faster performance than existing codes by leveraging improved optimal transport backends, including the exact network simplex implementation in `POT` [@flamary_pot_2021]. In addition, it offers an option to use sparse data structures, enabling the method to handle larger graphs efficiently.

The `pyotc` code provides two major approaches to the OTC problem. The *exact* solution procedure solves the underlying optimal transport problems via linear programming. Here the specialized *network simplex* algorithm is used from `POT` [@flamary_pot_2021] [@peyre_computational_2021], but we also provide a pure Python alternative. This is the core of the *improvement* step of policy iteration. In the *evaluation* step, we solve a block linear system involving the transition matrix $R$ and the cost vector $c$. Once policy iteration has fully converged, the stationary distribution of the resulting optimal transition coupling can then be computed. This can be approached in many ways, including as a spectral problem. For an interpretation and detailed discussion of the stationary distribution, see [@yi_alignment_2024].

Alternatively to this exact approach, `pyotc` also provides an iterative method based on entropic regularization. Here we provide solvers that leverage the extensive optimal transport capabilities in `POT` as well as custom implementations developed from scratch. The performance of these options is summarized in Table `\ref`.

<!--- 
Test algorithm notation for pandoc
-->
Algorithm 1 from [@oconnor_optimal_2022]
\begin{algorithm}[H]
\DontPrintSemicolon
\LinesNotNumbered 
\SetKwInOut{Input}{Input}\SetKwInOut{Output}{Output}
\Input{$R_0 = P \otimes Q$, $\tau$}
\Output{$R$ an optmial transition coupling}
\BlankLine
converged = False \;
R += [R] \;
i = 0 \;
\tcc{iterate until converged}
\While{not converged}{
    \tcc{Evaluate transition coupling/policy}
    g, h = evaluate(R) \;
    \tcc{Improve transition coupling/policy}
    R += [improve(g, h)] \;
    \tcc{Check convergence}
    d = $\|$R[i+1] - R[i]$\|$ \;
    converged = d < $\tau$ \;
}
\caption{Exact OTC Algorithm 1}
\end{algorithm}



# Examples
We provide a basic hello world example here. Our implementation is well documented and simple, consisting essentially of Python functions, which makes it easy to modify.

<!--- 
Below is a notional interface; this is still in process for our development.
-->
```python
from pyotc.otc_backend.policy_iteration.dense.exact import exact_otc
import numpy as np

P = np.array([[.5, .5], [.5, .5]])
Q = np.array([[0, 1], [1, 0]])
c = np.array([[1, 0], [0, 1]])

exp_cost, R, stat_dist = exact_otc(P, Q, c)
print("\nExact OTC cost between P and Q:", exp_cost)
```

# Conclusion
`pyotc` provides a performant Python implementation for computing optimal transition couplings for stationary Markov chains and their associated graph structures. Optimal transition coupling is a classic example of opportunity to bring a novel computational tool to wider audience through open source software and improve it. By moving to an open ecosystem such as Python, we have produced an OTC code that is faster and, arguably, more capable than existing implementations.

As OTC is an active research topic, we believe there are significant opportunities to extend the work here.
In this direction, we hope that this code will facilitate further explorations in both novel algorithms and more general implementations.
One could explore for example variations on the policy improvement and policy evaluation algorithms in terms of the stationary distribution (essentially a resolvent calculation).
Implementation-wise, there are significant opportunities to provide additional interfaces to Python ecosystem, for example interfaces chem or bio informatics sources (for example RDKit [@landrum_rdkitrdkit_2025])
`pyotc` also enables additional benchmarking studies.

# Acknowledgments

# References
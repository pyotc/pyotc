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
 - name: Department Statistics and Operations Research, University of North Carolina---Chapel Hill, Chapel Hill, NC
   index: 1
 - name: Boston University, Boston, NC
   index: 2
 - name: Applied Research Associates
   index: 3
date: 28 May 2024
bibliography: paper.bib
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
---

# Summary
Recent scholarly works [@oconnor_optimal_2022] have introduced an extension of optimal transport that applies directly to stationary Markkov Process.
This allows for one to compute a useful distance between these objects, which can be useful for comparisons of networks/graphs, for example those coming from chemistry, biology, social science, and beyond.
We provide a performant python implementation of this method [@oconnor_optimal_2022] and interfaces for related network/graph based problems [@netotc].
Our implementation is open source, tested, and integrates with python data science ecosystem.

# Statement of need
Optimal transport proven a valuable, practical, and natural tool in data science and machine learning. As a problem in calculus of variations, conventional optimal transportation has many possible generalizations. A natural extension beyond probablity distributions is processes. Recent has provided initial theory and algorithms; we aim to provide a practical implementation of this work which is open for community extension. Secondly, our implementation provides a backend framework that is in parity with capability available for conventional optimal transport --- thereby offering parallelism via cpu and gpu.
Finally, we provide initial and careful baselines of performance. We expect this tool to be practically
applied in forthcoming work work on networks in chemistry, neuroscience, and biology.

`pyotc` addresses the following technical needs: providing a python implementation, accelerating computation, expanding size of problem possible in terms of memory.
There exist two other Matlab codes for optimal transport coupling. 
These codes have served as inspiration, but retain common Matlab challenges: not open or free (though free alternatives exist) and incomplete ecosystem for data science.
Python is the de facto language for data science and thus it natural to choose it for implementation.
Being the de facto language for data science we can plug into the rich existing ecosystem.
Namely for this specific project we integrate with `POT` (Python Optimal Transport) and network/graph theory tools likes like `networkx`.
Follow on application could involve integration with standard tools for machine learning such as `scikit-learn`.

In terms of accelerating computation, we provide a comparison for our implementation where we have made various choices for compute or storage.
Where possible we try to compare with the existing Matlab codes.
This is comparison is available in Table `\ref`.

# Features
Our implementation provides the tools necessary to recreate the examples given in [@oconnor_optimal_2022] and [@netotc] using python.
Our implementation is faster than available codes through use of better underlying optimal transport code coming from both the exact *network simplex* code available in `POT` [@POT].
We provide sparse storage which allows for scaling to larger problems in terms of stochastic block models.

The `pyotc` code provides two major approaches to the OTC problem. 
An *exact* solution procedure in which underlying optimal transport problems are solved by finding exact soltions via linear programming.
Here the specialized *network simplex* algorithm is used from `POT` [@POT] [@cuturi], but we also provide a pure Python alternative.
This is the core of the *improvement* part of policy iteration.
In the *evaluation* step of policy iteration one must determine a stationary distribution.
This can be approached many ways including as spectral problem.

Alternatively to this exact approach, `pytoc` also provides an iterative approach based upon an entropic regularization.
Here we provide a solvers which exploit large catalog of optimal transport capabilities in `POT` and from scratch impelmentations.
There performance of these options is cataloged in Table `\ref`

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

We provide the basic hello world example here and a number of other examples which can be modified to explore the method.
Our implementation is well documented and simple, esssentially python functions, and therefore allows for easy modification.

# Examples
<!--- 
Below is a notional interface; this is still in process for our development.
-->
```python
from pyotc.exact import ExactOTC
import numpy as np

P = np.array([[.5, .5], [.5, .5]])
Q = np.array([[0, 1], [1, 0]])

exact_otc = ExactOTC(P=P, Q=Q)
# takes one step of evaluation and improvement
exact_otc.step()

# psuedo code above becomes
exact_otc.reset()
converged = false
tau = 0.0001
while not converged:
  # pyotc provides interface to step
  exact_otc.step()
  # user defines stopping criteria
  d = numpy.linalg.norm(exact_otc.R[-1] - exact_otc.R[-2])
  converged = d < converged

print(f"Optimal Transport Coupling is {exact_otc.R[-1]}")
```

# Conclusion
`pyotc` provides a performant Python code for computing optimal transport couplings for stationary Markov chains and their related graph structures.
Optimal transportation coupling is a classic example of opportunity to bring a novel computational tool to wider audience through open source software and improve it.
Here we have shown moving to more open ecosystems such as Python have produced a faster and arguably more capable OTC code.

As OTC is an active research topic, we believe there are significant opportunities to extend the work here.
In this direction, we hope that this code will facilitate further explorations in both novel algorithms and more general implementations.
One could explore for example variations on the policy improvement and policy evaluation algorithms in terms of the stationary distribution (essentially a resolvent calculation).
Implementation-wise, there are significant opportunities to provide additional interfaces to Python ecosystem, for example interfaces chem or bioinformatics sources.
`pyotc` also enables additional benchmarking studies.

# Acknowledgments

# References
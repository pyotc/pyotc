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
Recent scholarly works [@oconnor_optimal_2022] have introduced an extension of optimal transport that applies directly to product structued processes.

# Statement of need
Optimal transport proven a valuable, practical, and natural tool in data science and machine learning. As a problem in calculus of variations, conventional optimal transportation has many possible generalizations. A natural extension beyond probablity distributions is processes. Recent has provided initial theory and algorithms; we aim to provide a practical implementation of this work which is open for community extension. Secondly, our implementation provides a backend framework that is in parity with capability available for conventional optimal transport --- thereby offering parallelism via cpu and gpu.
Finally, we provide initial and careful baselines of performance. We expect this tool to be practically
applied in forthcoming work work on networks in chemistry, neuroscience, and biology.

# Features
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

# Acknowledgments

# References
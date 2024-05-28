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
exact_otc.step()
```

# Conclusion

# Acknowledgments

# References
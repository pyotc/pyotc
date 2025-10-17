# pyotc
[![codecov](https://codecov.io/github/pyotc/pyotc/graph/badge.svg?token=52QPNW0AP7)](https://codecov.io/github/pyotc/pyotc)

A python implementations of optimal transport coupling algorithms.

## Documentation
Find sphinx documentation [here](https://pyotc.github.io/pyotc/).

## Installation

We expect `pyotc` to be pip-installable across all platforms. 


### 1. Install from pypi (Recommended)

```bash
pip install pyotc
```
- Note: `pyotc` requires Python 3.10 or above.

### 2. Install from github

```bash
pip install https://github.com/pyotc/pyotc.git
```

### 3. Install for Development
We test in venvs provided by [uv](https://docs.astral.sh/uv/) via [nox](https://nox.thea.codes/en/stable/usage.html#changing-the-sessions-default-backend). It's helpful, but not strictly necessary to do the same.

```bash
git clone https://github.com/pyotc/pyotc.git
cd pyotc
pip install -e .
```

### `uv` workflow
Install the [uv tool](https://docs.astral.sh/uv/getting-started/installation/). Then

```bash
git clone https://github.com/pyotc/pyotc.git
cd pyotc
uv sync
uv pip install -e .
```

To verify your installation, run
```bash
uv run pytest
```

## Run Tests
With a `uv` setup one can simply
```bash
uv run pytest
```
Otherwise, in `pip` installed context with deps met, `pytest` should behave as expected.

## Contributing
Guidelines for contribution to `pyotc` are provided in [CONTRIBUTING.md](./CONTRIBUTING.md).

## Changelog
A summary of changes and guide to versioning are recoreded in [CHANGELOG.md](./CHANGELOG.md).

## Citing this Repository
If you wish to cite our work, please use the following BibTeX code:
```
@article{yi2025alignment,
  title={Alignment and comparison of directed networks via transition couplings of random walks},
  author={Yi, Bongsoo and O'Connor, Kevin and McGoff, Kevin and Nobel, Andrew B},
  journal={Journal of the Royal Statistical Society Series B: Statistical Methodology},
  pages={qkae085},
  year={2024},
  doi = {10.1093/jrsssb/qkae085}
}
```
# Install
We expect pyotc to be pip-installable across all platforms.

## Install for usage
### From pypi

```pip install pyotc```
[TODO#26](https://github.com/jhineman/pyotc/issues/26)

### From github

```pip install https://github.com/jhineman/pyotc.git```

## Install for development
We test in venvs provided by [uv](https://docs.astral.sh/uv/) via [nox](https://nox.thea.codes/en/stable/usage.html#changing-the-sessions-default-backend). It's helpful, but not strictly necessary to do the same.

```bash
git clone https://github.com/jhineman/pyotc.git
cd pyotc
pip install -e .
```

### `uv` workflow
[Install the uv tool](https://docs.astral.sh/uv/getting-started/installation/).

Then

```bash
git clone https://github.com/jhineman/pyotc.git
cd pyotc
uv sync
```





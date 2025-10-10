# Release Guide

This document describes the steps to create a new tagged release of the project.

## When to version
We use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Semantic versioning provides both convenient points in history and summarizes the types of changes that were made.

A *fast and loose* guide to this is as follows:
   - **MAJOR**: incompatible API changes.
   - **MINOR**: backwards-compatible features.
   - **PATCH**: backwards-compatible bug fixes.

We aim to *limit* **MAJOR** changes and thereby preserve API compatibility.

## Release Checklist
Before cutting a release, ensure:

- [ ] All tests pass on the `main` branch (CI green).
- [ ] Dependencies are up to date (see [CONTRIBUTING.md](./CONTRIBUTING.md))
- [ ] Documentation is updated if necessary.
- [ ] `CHANGELOG.md` has been updated with all user-facing changes.
- [ ] Version numbers are consitent across `pyproject.toml`, `CHANGELOG.md`, and elsewhere.

### Hint on the Checklist
1. Version can be checked and updated with recent releases of `uv` with `uv version`. Read more with `uv version --help`

## Doing the release
Once `main` meets the checks above create a new tagged release using the [releases dialog](https://github.com/pyotc/pyotc/releases/new).

## Building wheels and publishing them on pypi
We use `uv` to do this. The details of how are in the uv documentation on [building and publishing a package](https://docs.astral.sh/uv/guides/package/#building-and-publishing-a-package).

### Testing a published release on pypi
The test index `testpypi` is configured in the [pyproject.toml](./pyproject.toml) and can be used for testing with:
```bash
uv publish --index testpypi
```

### Automated deployment
The automation of this deployment is done via [github actions](https://github.com/pypa/gh-action-pypi-publish).

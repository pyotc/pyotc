# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2025-10-10
* Add github actions build wheels and upload them to pypi.

## [0.2.1] - 2025-10-10
* Fix verion in `pyproject.toml`

## [0.2.0] - 2025-10-02

### Added
* [RELEASE.md](./RELEASE.md)
* Install with `uv`
* `pytest` testing
* `nox` for linting (`ruff`), formating (`ruff`), and testing (`pytest`)
* [sphinx documentation](https://pyotc.github.io/pyotc/)
* github actions running `nox` and generating documentation
* sparse versions of exact otc
* entropic otc

### Changed
* Use `md` vs `rst` in most cases

### Removed
* makefiles from cookie cutter

## [0.1.0] - 2024-02-27

### Added
* Initial cookiecutter template
* Basic algorithm

[unreleased]: https://github.com/pyotc/pyotc
[0.1.0]: https://github.com/pyotc/pyotc/tree/v0.1.0

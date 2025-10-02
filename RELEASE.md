# Release Guide

This document describes the steps to create a new tagged release of the project.

---

## Pre-Release Checklist

Before cutting a release, ensure:

- [ ] All tests pass on the `main` branch (CI green).
- [ ] Dependencies are up to date (see [CONTRIBUTING.md](./CONTRIBUTING.md))
- [ ] Documentation is updated if necessary.
- [ ] `CHANGELOG.md` has been updated with all user-facing changes.
- [ ] Version numbers are consitent across `pyproject.toml`, `CHANGELOG.md`, and elsewhere.

---

## When to version
We use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Semantic versioning provides both convenient points in history and summarizes the types of changes that were made.

A *fast and loose* guide to this is as follows:
   - **MAJOR**: incompatible API changes.
   - **MINOR**: backwards-compatible features.
   - **PATCH**: backwards-compatible bug fixes.

We aim to *limit* **MAJOR** changes and thereby preserve API compatibility.

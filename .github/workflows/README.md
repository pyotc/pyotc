# GitHub Actions Workflows

This directory contains YAML workflows for GitHub Actions automation.

## Workflows Overview
- [build_and_publish.yml](build_and_publish.yml) – Deploys the project to production when a release is created.
- [lint_and_test.yml](lint_and_test.yml) – Runs nox to check (ruff), format (ruff), and test (pytest) for a matrix of pythons.
- [sphinx.yml](sphinx.yml) - Run sphinx documentation and deploys to github pages.


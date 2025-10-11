# GitHub Actions Workflows

This directory contains YAML workflows for GitHub Actions automation.

## Workflows Overview
- [build_and_test.yml](build_and_test.yml) – Runs nox to check (ruff), format (ruff), and test (pytest) for a matrix of pythons, and then builds wheels.
- [sphinx.yml](sphinx.yml) - Run sphinx documentation and deploys to github pages.
- [publish.yml](publish.yml) – Deploys the project to production when a release is created.

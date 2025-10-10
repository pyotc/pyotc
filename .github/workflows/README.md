# GitHub Actions Workflows

This directory contains YAML workflows for GitHub Actions automation.

## Workflows Overview
- [nox.yml](nox.yml) – Runs nox to check (ruff), format (ruff), and test (pytest) pull requests.
- [sphinx.yml](sphinx.yml) - Run sphinx documentation and deploys to github pages.
- [build_wheels.yml](build_wheels.yml) - Use CI Build Whell to build wheels for multiple Linux/Mac/Windows
- [publish.yml](publish.yml) – Deploys the project to production when a release is created.

"""`noxfile` for pyotc.

Provides parametrized linting, formating, and testing using ruff and pytest
"""

import nox
from nox import session

nox.options.default_venv_backend = "uv|virtualenv"

py_versions = ["3.12", "3.11", "3.10"]


@session
def lint(session):
    session.install("ruff")
    session.run("ruff", "check", "--show-files")


@session
def format_check(session):
    session.install("ruff")
    session.run("ruff", "format", "--check")


@session(python=py_versions)
def tests(session):
    session.install(".")
    session.install("pytest", "pytest-cov")
    session.run("pytest --cov --cov-branch --cov-report=xml")

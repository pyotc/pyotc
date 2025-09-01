"""`noxfile` for pyotc.

Provides linting, formating, and testing using ruff and pytest
"""

import nox
from nox import session

nox.options.default_venv_backend = "uv|virtualenv"


@session
def lint(session):
    session.install("ruff")
    session.run("ruff", "check", "--show-files")


@session
def format_check(session):
    session.install("ruff")
    session.run("ruff", "format", "--check")


@session
def tests(session):
    session.install(".")
    session.install("pytest", "pytest-cov")
    session.run("pytest")

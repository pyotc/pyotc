# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at [https://github.com/jhineman/pyotc/issues](https://github.com/jhineman/pyotc/issues).

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

See improvements listed in the [issues](https://github.com/jhineman/pyotc/issues).

### Write Documentation

`pyotc` could always use more documentation, whether as part of the official `pyotc` docs, in docstrings, or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at [https://github.com/jhineman/pyotc/issues](https://github.com/jhineman/pyotc/issues).

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are welcome! :)

## Get Started!

Ready to contribute? Here's how to set up `pyotc` for local development.

1. Fork the `pyotc` repo on GitHub.
2. Clone your fork locally:

    ```shell
    git clone git@github.com:your_name_here/pyotc.git
    ```

3. Install your local copy into a virtualenv. See the [install directions](INSTALL.md)

4. Create a branch for local development:

    ```shell
    git checkout -b name-of-your-bugfix-or-feature
    ```

   Now you can make your changes locally.

5. When you're done making changes use [nox](nox) to lint, format, and test.

6. Commit your changes and push your branch to GitHub:

    ```shell
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature
    ```

7. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

0. Run `nox` in the root directory. Other [nox cli](https://nox.thea.codes/en/stable/usage.html#command-line-usage) options are avaiable.
1. The pull request should include tests for new functionality.
2. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring, and add the feature to the list in `README.md`.
3. We use github actions ([TODO#19](https://github.com/jhineman/pyotc/issues/19)) for our CI which runs on nox.

## `uv` workflow

### Adding dependencies with uv
If you're adding true dependency, say for example `pytorch`, this is done simply with
```bash
uv add pytorch
```
See also documentation on [adding dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/#adding-dependencies)

If you're adding a [development dependency](https://docs.astral.sh/uv/concepts/projects/dependencies/#development-dependencies) (e.g `pytest`) there is a little extra
```bash
uv add --dev pytest
```

### Running nox via `uv`
```bash
# in project root
uv run nox
```
### Running ruff format via `uv`
```bash
# in project root
uv run ruff format
```


# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at [https://github.com/pyotc/pyotc/issues](https://github.com/pyotc/pyotc/issues).

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

See improvements listed in the [issues](https://github.com/pyotc/pyotc/issues).

### Write Documentation

`pyotc` could always use more documentation, whether as part of the official `pyotc` docs, in docstrings, or even on the web in blog posts, articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at [https://github.com/pyotc/pyotc/issues](https://github.com/pytoc/pyotc/issues).

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

0. Run `nox` in the root directory. Other [nox cli](https://nox.thea.codes/en/stable/usage.html#command-line-usage) options are avaiable. We use this same workflow in github actions. This should install, lint, format, and test your code.
1. Include new tests for new functionality.
2. Include new documentation for new functionality.
3. Include updates to the [changelog](./CHANGELOG.md).
4. Prepare to [help other review your changes](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/helping-others-review-your-changes).

## `uv` workflow

### Adding dependencies with `uv`
If you're adding true dependency, say for example `pytorch`, this is done simply with
```bash
uv add pytorch
```
See also documentation on [adding dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/#adding-dependencies)

If you're adding a [development dependency](https://docs.astral.sh/uv/concepts/projects/dependencies/#development-dependencies) (e.g `pytest`) there is a little extra
```bash
uv add --dev pytest
```

### Running nox via `uv` as tool
```bash
# in project root
uv tool run nox
```
*Note that this uses `nox` in isolation and should mimic what is done in [github actions](.github/workflows/nox.yml)*

### Running ruff format via `uv`
```bash
# in project root
uv tool run ruff format
```
*Note that this uses `ruff` in isolation and should mimic what is done in [github actions](.github/workflows/nox.yml)*
*Ruff in particular on your system, vs as tool, may be divergent.*


### Tagged releases
We adhere to the following release process described in the template available in [RELEASE.md](RELEASE.md).
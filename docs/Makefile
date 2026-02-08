# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = pyotc
SOURCEDIR     = .
BUILDDIR      = _build

# Custom pre-processing: sync INSTALL.md before building HTML
html:
	@echo "Syncing CONTRIBUTING.md to docs/CONTRIBUTING.md..."
	cp ../CONTRIBUTING.md CONTRIBUTING.md
	@echo "Syncing CHANGELOG.md to docs/CHANGELOG.md..."
	cp ../CHANGELOG.md CHANGELOG.md
	@echo "Syncing RELEASE.md to docs/RELEASE.md..."
	cp ../RELEASE.md RELEASE.md
	@echo "Syncing pyproject.toml to docs/pyproject.toml..."
	cp ../pyproject.toml pyproject.toml
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# %: Makefile
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

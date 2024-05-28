# JOSS notes

* [Paper Format](https://joss.readthedocs.io/en/latest/paper.html#joss-paper-format)
* [Checking Compile](https://joss.readthedocs.io/en/latest/paper.html#checking-that-your-paper-compiles)
* [Quick checks with pandoc for the impatient](https://gist.github.com/dcchambers/9761c71880114cc604c902b30b2e06c8)

## Pandoc commmand
In this directory
```bash
pandoc -o paper.pdf -M link-citations=true --citeproc --bibliography=paper.bib paper.md
```
builds the paper draft. *Note this not formatted, but can be used for quick-and-dirty iteration.*
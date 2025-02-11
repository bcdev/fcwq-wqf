# Contributing

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
## Coding standards

Coding standards applied and adhered to in this project are based on [PEP 8](https://pep8.org) 
guidelines. [Black](https://github.com/psf/black) is used to ensure that coding standards are met.

## Version control of Jupyter notebooks

To version-control Jupyter notebooks avoiding that each run is interpreted as
new version of the notebook, `cd` into the project root directory and type

    nbstripout --install

This ensures that whenever you commit a Jupyter notebook to the repo, the output
will be stripped automatically. The result is that only the code and  markdown
are versioned.

## How to create a new release?

1. Run all unit tests. No test shall fail.
2. ...
3. Commit and push all changes.
4. Create a new tag `vYYYY.D.D` on [GitHub](https://github.com/bcdev/fcwq). 
5. Create a new release branch `release/vYYYY.D.D` from the new tag.

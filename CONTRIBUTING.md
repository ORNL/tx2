# Contributing to TX2

Help in improving TX2 is welcome! 

If you find a bug or think of an enhancement/improvement you would like to see,
feel free to fill out an appropriate
[issue](https://github.com/ORNL/tx2/issues/new/choose).

If you have a question, double check that it's not covered in our
[documentation](https://ornl.github.io/tx2).

For questions not answered by the docs or anything else that might not fit into one
of the issue templates, you can start a discussion in the [dicussions
tab](https://github.com/ORNL/tx2/discussions).

You are also welcome to contact the developers directly by emailing us at
tx2-help@ornl.gov.

## Submitting a PR

If you have added a useful feature or fixed a bug, open a new pull request with
the changes.  When submitting a pull request, please describe what the pull 
request is addressing and briefly list any significant changes made. If it's in
regards to a specific issue, please include the issue number. Please check and
follow the formatting conventions below!

## Code Formatting

This project uses the [black code formatter](https://github.com/psf/black).

Any public functions and clases should be clearly documented with 
[sphinx-style docstrings](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html).
Local documentation can be generated with

```bash
cd docs
make html
```

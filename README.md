# TX2

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/tx2.svg)](https://badge.fury.io/py/tx2)
[![JOSS status](https://joss.theoj.org/papers/b7c161917e5a31af052a597bf98f0e94/status.svg)](https://joss.theoj.org/papers/b7c161917e5a31af052a597bf98f0e94)
[![tests](https://github.com/ORNL/tx2/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/ORNL/tx2/actions/workflows/tests.yml)

Welcome to TX2! This library is intended to aid in the explorability and explainability of
transformer classification networks, or transformer language models with sequence classification
heads. The basic function of this library is to take a trained transformer and
test/train dataset and produce an ipywidget dashboard as seen in the screenshot below,
which can be displayed in a jupyter notebook or jupyter lab.

![screenshot]( https://raw.githubusercontent.com/ORNL/tx2/master/sphinx/source/screenshot.png)

NOTE: Currently this library's implementation is partially torch-dependent, and so will
not work with tensorflow/keras models - we hope to address this limitation in the future!

## Installation

You can install this package from pypi:

```bash
pip install tx2
```

NOTE: depending on the environment, it may be better to install some of the dependencies separately before
pip installing tx2, e.g. in conda:
```bash
conda install pytorch-gpu pandas scikit-learn matplotlib ipywidgets "numpy<=1.20" -c conda-forge
```

If you do not have access to a GPU on your machine, install the regular pytorch
package:
```bash
conda install pytorch pandas scikit-learn matplotlib ipywidgets "numpy<=1.20"
```

Note that `pytorch-gpu` can only be found in the `conda-forge` channel.

## Examples

Example jupyter notebooks demonstrating and testing the usage of this library can be
found in the [examples
folder](https://github.com/ORNL/tx2/tree/master/examples).

Note that these notebooks can take a while to run the first time, especially
if a GPU is not in use.

Packages you'll need to install for the notebooks to work (in addition to the
conda installs above):

```bash
pip install tqdm transformers~=4.1.1
```

Running through each full notebook will produce the ipywidget dashboard near the
end.

The tests in this repository do not depend on transformers, so raw library
functionality can be tested by running `pytest` in the project root.

## Documentation

The documentation can be viewed at [https://ornl.github.io/tx2/](https://ornl.github.io/tx2/).

The documentation can also be built from scratch with sphinx as needed.

Install all required dependencies:
```bash
pip install -r requirements.txt
```

Build documentation:

```bash
cd docs
make html
```

The `docs/build/html` folder will now contain an `index.html`

Two notebooks demonstrating the dashboard and how to use TX2 are included
in the `examples` folder, highlighting the default and custom approaches
as discussed in the Basic Usage page of the documentation.

## Citation

To cite usage of TX2 in a publication, the DOI for this code is [https://doi.org/10.21105/joss.03652](https://doi.org/10.21105/joss.03652)

bibtex:
```
@article{Martindale2021,
  doi = {10.21105/joss.03652},
  url = {https://doi.org/10.21105/joss.03652},
  year = {2021},
  publisher = {The Open Journal},
  volume = {6},
  number = {68},
  pages = {3652},
  author = {Nathan Martindale and Scott L. Stewart},
  title = {TX$^2$: Transformer eXplainability and eXploration},
  journal = {Journal of Open Source Software}
}
```

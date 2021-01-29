.. TX2 documentation master file, created by
   sphinx-quickstart on Wed Nov 25 09:02:08 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. TODO -  Visualization. Utils. README

TX\ :sup:`2` Documentation
===============================

Welcome to TX\ :sup:`2`! This library is intended to aid in the explorability and explainability of
transformer classification networks, or transformer language models with sequence classification
heads. The basic function of this library is to take a trained transformer and
test/train dataset and produce an ipywidget dashboard as seen in the screenshot below,
which can be displayed in a jupyter notebook or jupyter lab.

.. image:: screenshot.png

NOTE: Currently this library's implementation is partially torch-dependent, and so will
not work with tensorflow/keras models - we hope to address this limitation in the future!

.. toctree::
   :maxdepth: 2
   :caption: Usage

   basic_usage.rst
   dashboard_widgets.rst

.. toctree::
   :maxdepth: 2
   :caption: API

   wrapper.rst
   dashboard.rst
   calc.rst
   visualization.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

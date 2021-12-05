========================
Bayesian target encoding
========================

``bayte`` offers a lightweight, ``scikit-learn``-compliant\ :footcite:p:`sklearn_api`
implementation of Bayesian Target Encoding. The algorithm was introduced in 2019 by
:footcite:t:`slakey`, with ensemble modeling methodology from :footcite:t:`larionov`.
Our explanation of the algorithm is available :doc:`here <explanation/algorithm>`.

Installation
------------

To install ``bayte`` from PyPI, run

.. code-block:: console

    $ python -m pip install bayte

This is the preferred method to install ``bayte``.

Contents
--------

.. toctree::
    :maxdepth: 2
    :caption: Quickstart

    experiments/index

.. toctree::
    :maxdepth: 2
    :caption: How-to guides

    how-to/encode
    how-to/estimate

.. toctree::
    :maxdepth: 2
    :caption: Explanation

    explanation/algorithm

.. toctree::
    :maxdepth: 2
    :caption: Reference
    
    GitHub repository <https://github.com/ak-gupta/bayte>

.. footbibliography::

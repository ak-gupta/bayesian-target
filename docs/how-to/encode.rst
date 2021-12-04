=================================
Encode your categorical variables
=================================

To encode your variables, you have to first choose a likelihood for your target.

+----------------+-------------+-----------------+
| Model type     | Description | Likelihood      |
|                |             |                 |
+================+=============+=================+
| Classification | Binary      | ``bernoulli``   |
+----------------+-------------+-----------------+
|                | Multi-class | ``multinomial`` |
+----------------+-------------+-----------------+
| Regression     |             | ``exponential`` |
+----------------+-------------+-----------------+
|                |             | ``gamma``       |
+----------------+-------------+-----------------+
|                |             | ``invgamma``    |
+----------------+-------------+-----------------+

Basic usage
-----------

Once you've chosen your likelihood, import and fit the encoder on your data. Suppose
you have ``X`` and ``y``, with three categorical columns: 1, 2, and 5.

.. code-block:: python
    :emphasize-lines: 3

    import bayte as bt

    encoder = bt.BayesianTargetEncoder(dist=...)
    encoder.fit(X[:, [1, 2, 5]], y)

By default, when you transform the data

.. code-block:: python

    X_encoded = encoder.transform(X[:, [1, 2, 5]])

the encoding level will be the mean of the posterior distribution for the level.
To sample, set ``sample=True`` on encoder initialization.

.. important::

    The encoder has support for `joblib <https://scikit-learn.org/stable/computing/parallelism.html>`_.
    Since the encoding procedure involves generating posterior parameters for every categorical level in
    every supplied variable, it can be computationally inefficient if executed serially.

Changing hyperparameter initialization
--------------------------------------

If you want to change how the hyperparameters are initialized for a given likelihood,
supply a callable for the ``initializer`` argument. This callable must take the ``dist``
and the target values ``y`` and return a tuple of the parameters.

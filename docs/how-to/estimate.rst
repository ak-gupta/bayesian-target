=======================================
Build a model through repeated encoding
=======================================

In this guide, we will build an ensemble estimator using bayesian target encoding.
First, initialize the ``bayte`` class for your modelling problem your base estimator
as well as an initialized encoder and the number of samples you want to draw.

.. tabs::

    .. tab:: Classification

        In this classification example, we will fit a logistic regression.

        .. code-block:: python
            :emphasize-lines: 8

            from sklearn.linear_model import LogisticRegression

            import bayte as bt

            estimator = bt.BayesianTargetClassifier(
                base_estimator=LogisticRegression(),
                encoder=bt.BayesianTargetEncoder(dist="bernoulli"),
                n_estimators=10,
            )

    .. tab:: Regression

        In this regression example, we will fit a simple linear model.

        .. code-block:: python
            :emphasize-lines: 8

            from sklearn.linear_model import LinearRegression

            import bayte as bt

            estimator = bt.BayesianTargetRegressor(
                base_estimator=LinearRegression(),
                encoder=bt.BayesianTargetEncoder(dist="gamma"),
                n_estimators=10,
            )

Next, call ``fit``. The ``fit`` call accepts a ``categorical_feature`` parameter for
specifying which features in the training dataset should be encoded. If you are using a
``numpy.ndarray``, specify the indices of the categorical columns. If you are using a
``pandas.DataFrame``, specify the column names. If not set, any pandas columns with a
categorical data type will be encoded; for numpy users, no supplied list indicates
that *all* features are categorical.

.. code-block:: python

    estimator.fit(X, y, categorical_feature=[1, 2, 5])

For regression problems, ``predict`` will produce an average of each estimator prediction.
For classification, ``predict`` depends on an estimator initialization parameter called
``voting``\ :footcite:p:`scikit-learn`:

+--------------------+-------------------------------------------------------+
| ``voting`` value   | Description                                           |
|                    |                                                       |
+====================+=======================================================+
| ``hard`` (default) | | The predicted class label is a majority vote of the |
|                    | | predicted labels from each estimator.               |
+--------------------+-------------------------------------------------------+
| ``soft``           | | The predicted class label is based on the sums of   |
|                    | | predicted probabilities for each class.             |
+--------------------+-------------------------------------------------------+

For well-calibrated classifiers, ``soft`` voting is preferred.

.. footbibliography::

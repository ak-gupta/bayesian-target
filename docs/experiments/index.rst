=================================
When should you use this package?
=================================

Coming soon.

The plan is to leverage the experimental framework discussed by :footcite:t:`pargent`
to analyze bayesian target encoding (BTE) and answer the following questions:

- How does the effectiveness of BTE change with the number of categorical levels?

  - Since it's hard to compare effectiveness of encoding approaches across
    different datasets, I will likely need to use synthetic data to answer
    this question.

- **Marginal BTE**: Is there lift from a staged approach:

  #. Fit a submodel [*]_ that uses all non-categorical columns to predict the target.
  #. Fit the encoder using the submodel output as the target [*]_.
  #. Use the encoding and the raw input non-categorical data to fit the final model.

- Ensemble methodology\ :footcite:p:`larionov`

  - How much does repeated sampling help?
  - How many samples do you need?

.. [*] Does the submodel algorithm matter?
.. [*]

    What if the encoder is fitted using the residuals from the submodel as the
    target?

Comparative encoding methodology
--------------------------------

When conducting these experiments, we'll compare BTE to the following encoding
methodologies. Suppose you have :math:`n` training observations, with
:math:`Y = (y_{1}, ..., y_{n})` representing the target and categorical variable
:math:`X_{1} = (x_{1}, ..., x_{n})` with distinct values :math:`V = (v_{1}, ..., v_{l})`.

:footcite:t:`pargent` provide a description for each encoding methodology listed
below.

+--------------------------------+-------------+------------------------------------------+
| Encoding                       | Supervised? | Implementation                           |
|                                |             |                                          |
+================================+=============+==========================================+
| Frequency                      | N           | ``category_encoders.CountEncoder``       |
+--------------------------------+-------------+------------------------------------------+
| Generalized Linear Mixed Model | Y           | ``category_encoders.GLMMEncoder``        |
+--------------------------------+-------------+------------------------------------------+
| James-Stein                    | Y           | ``category_encoders.JamesSteinEncoder``  |
+--------------------------------+-------------+------------------------------------------+
| One-hot                        | N           | ``sklearn.preprocessing.OneHotEncoder``  |
+--------------------------------+-------------+------------------------------------------+
| Integer                        | N           | ``sklearn.preprocessing.OrdinalEncoder`` |
+--------------------------------+-------------+------------------------------------------+
| Target                         | Y           | ``category_encoders.TargetEncoder``      |
+--------------------------------+-------------+------------------------------------------+

Modeling algorithms
-------------------

The following modelling implementations will be tested:

+------------------------------------------+------------------------+
| Package                                  | Class                  |
|                                          |                        |
+==========================================+========================+
| LightGBM\ :footcite:p:`lightgbm`         | ``LGBMClassifier``     |
+------------------------------------------+------------------------+
|                                          | ``LGBMRegressor``      |
+------------------------------------------+------------------------+
| Scikit-Learn\ :footcite:p:`scikit-learn` | ``LinearRegression``   |
+------------------------------------------+------------------------+
|                                          | ``LogisticRegression`` |
+------------------------------------------+------------------------+
| XGBoost\ :footcite:p:`xgboost`           | ``XGBClassifier``      |
+------------------------------------------+------------------------+
|                                          | ``XGBRegressor``       |
+------------------------------------------+------------------------+

Datasets
--------

Below is a list of the regression datasets used for
experimentation\ :footcite:p:`pargent`.

+-------------------------------------------+---------------------------------------------------------------+
| OpenML ID                                 | Dataset name                                                  |
|                                           |                                                               |
+===========================================+===============================================================+
| `41211 <https://www.openml.org/d/41211>`_ | :doc:`ames-housing <regression/housing>`                      |
+-------------------------------------------+---------------------------------------------------------------+
| `41445 <https://www.openml.org/d/41445>`_ | :doc:`employee_salaries <regression/salaries>`                |
+-------------------------------------------+---------------------------------------------------------------+
| `41210 <https://www.openml.org/d/41210>`_ | :doc:`avocado-sales <regression/avocado>`                     |
+-------------------------------------------+---------------------------------------------------------------+
| `41267 <https://www.openml.org/d/41267>`_ | :doc:`particulate-matter-ukair-2017 <regression/particulate>` |
+-------------------------------------------+---------------------------------------------------------------+
| `41251 <https://www.openml.org/d/41251>`_ | :doc:`flight-delay-usa-dec-2017 <regression/flight>`          |
+-------------------------------------------+---------------------------------------------------------------+
| `41255 <https://www.openml.org/d/41255>`_ | :doc:`nyc-taxi-green-dec-2016 <regression/taxi>`              |
+-------------------------------------------+---------------------------------------------------------------+

Performance evaluation
----------------------

:footcite:t:`pargent` discussed a three-phase approach for creating a baseline
assessment of model performance. We'll adapt that here and use something slightly
different: 

  **baseline performance** is the average test score for a model fitted with
  **no categorical features** using 5-fold cross-validation.

Similar to :footcite:t:`pargent`, we will use root mean squared error (RMSE) for
evaluating the performance of regression models and the area under the receiver
operating characteristic (AUROC) for classification problems. Both metrics are
available in ``scikit-learn``\ :footcite:p:`scikit-learn` under the strings
``neg_root_mean_squared_error`` and ``roc_auc``, respectively.

.. footbibliography::

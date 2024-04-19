"""Test the ensemble estimator.


NOTE: Since we can't control the ``categorical_feature`` input to ``fit`` in the
``sklearn.utils.estimator_checks.check_estimator`` call, we aren't running the test.
"""

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted

import bayte as bt


def test_estimator_reg_fit(toy_regression_dataset):
    """Test a basic fit."""
    X, y = toy_regression_dataset
    estimator = bt.BayesianTargetRegressor(
        estimator=SVR(kernel="linear"),
        encoder=bt.BayesianTargetEncoder(dist="normal"),
        n_estimators=2,
    )
    estimator.fit(
        X,
        y,
        categorical_feature=[
            9,
        ],
    )

    assert hasattr(estimator, "estimators_")
    assert len(estimator.estimators_) == 2
    assert not np.array_equal(
        estimator.estimators_[0].coef_, estimator.estimators_[1].coef_
    )


def test_estimator_parallel_fit(toy_regression_dataset):
    """Test a parallel fit."""
    X, y = toy_regression_dataset
    estimator = bt.BayesianTargetRegressor(
        estimator=SVR(kernel="linear"),
        encoder=bt.BayesianTargetEncoder(dist="normal"),
        n_estimators=2,
        n_jobs=2,
    )
    estimator.fit(
        X,
        y,
        categorical_feature=[
            9,
        ],
    )

    assert hasattr(estimator, "estimators_")
    assert len(estimator.estimators_) == 2
    for est in estimator.estimators_:
        check_is_fitted(est)

    assert not np.array_equal(
        estimator.estimators_[0].coef_, estimator.estimators_[1].coef_
    )


def test_estimator_clf_fit(toy_classification_dataset):
    """Test a basic fit with a classification task."""
    X, y = toy_classification_dataset
    estimator = bt.BayesianTargetClassifier(
        estimator=LogisticRegression(),
        encoder=bt.BayesianTargetEncoder(dist="bernoulli"),
        n_estimators=2,
    )
    estimator.fit(
        X,
        y,
        categorical_feature=[
            9,
        ],
    )

    assert hasattr(estimator, "estimators_")
    assert len(estimator.estimators_) == 2
    for est in estimator.estimators_:
        check_is_fitted(est)
    assert not np.array_equal(
        estimator.estimators_[0].coef_, estimator.estimators_[1].coef_
    )


def test_estimator_fit_pandas(toy_regression_dataset):
    """Test a basic fit with a pandas DataFrame."""
    X, y = toy_regression_dataset
    estimator = bt.BayesianTargetRegressor(
        estimator=SVR(kernel="linear"),
        encoder=bt.BayesianTargetEncoder(dist="normal"),
        n_estimators=2,
    )
    X = pd.DataFrame(X)
    X[9] = X[9].astype("category")

    estimator.fit(X, y)

    assert_array_equal(
        estimator.categorical_,
        np.array([False, False, False, False, False, False, False, False, False, True]),
    )
    assert hasattr(estimator, "estimators_")
    assert len(estimator.estimators_) == 2
    assert not np.array_equal(
        estimator.estimators_[0].coef_, estimator.estimators_[1].coef_
    )


def test_estimator_fit_pandas_manual(toy_regression_dataset):
    """Test a basic fit with a pandas DataFrame and no automatic detection."""
    X, y = toy_regression_dataset
    estimator = bt.BayesianTargetRegressor(
        estimator=SVR(kernel="linear"),
        encoder=bt.BayesianTargetEncoder(dist="normal"),
        n_estimators=2,
    )
    X = pd.DataFrame(X)
    X.columns = X.columns.astype(str)

    estimator.fit(
        X,
        y,
        categorical_feature=[
            "9",
        ],
    )

    assert_array_equal(
        estimator.categorical_,
        np.array([False, False, False, False, False, False, False, False, False, True]),
    )
    assert hasattr(estimator, "estimators_")
    assert len(estimator.estimators_) == 2
    assert not np.array_equal(
        estimator.estimators_[0].coef_, estimator.estimators_[1].coef_
    )


def test_estimator_reg_prefit(toy_regression_dataset):
    """Test a basic fit with a pre-fitted encoder."""
    X, y = toy_regression_dataset
    encoder = bt.BayesianTargetEncoder(dist="normal")
    encoder.fit(X[:, [9]], y)

    estimator = bt.BayesianTargetRegressor(
        estimator=SVR(kernel="linear"),
        encoder=encoder,
        n_estimators=2,
    )
    estimator.fit(
        X,
        y,
        categorical_feature=[
            9,
        ],
    )

    assert hasattr(estimator, "estimators_")
    assert len(estimator.estimators_) == 2
    assert not np.array_equal(
        estimator.estimators_[0].coef_, estimator.estimators_[1].coef_
    )


def test_estimator_reg_predict(toy_regression_dataset):
    """Test basic prediction with a regression dataset."""
    X, y = toy_regression_dataset
    estimator = bt.BayesianTargetRegressor(
        estimator=SVR(kernel="linear"),
        encoder=bt.BayesianTargetEncoder(dist="normal"),
        n_estimators=2,
    )
    estimator.fit(
        X,
        y,
        categorical_feature=[
            9,
        ],
    )

    y = estimator.predict(X)

    assert y.shape == (1000,)


def test_estimator_clf_predict(toy_classification_dataset):
    """Test basic prediction with a classification target."""
    X, y = toy_classification_dataset
    estimator = bt.BayesianTargetClassifier(
        estimator=LogisticRegression(),
        encoder=bt.BayesianTargetEncoder(dist="bernoulli"),
        n_estimators=10,
    )
    estimator.fit(
        X,
        y,
        categorical_feature=[
            9,
        ],
    )

    y = estimator.predict(X)
    yprob = estimator.predict_proba(X)

    assert y.shape == (1000,)
    assert_array_equal(np.unique(y), np.arange(2))
    assert yprob.shape == (1000, 2)
    assert ((yprob > 1) & (yprob < 0)).sum() == 0

    estimator.set_params(voting="soft")

    y = estimator.predict(X)

    assert y.shape == (1000,)
    assert_array_equal(np.unique(y), np.arange(2))

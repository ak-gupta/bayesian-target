"""Test the ensemble estimator.


NOTE: Since we can't control the ``categorical_feature`` input to ``fit`` in the
``sklearn.utils.estimator_checks.check_estimator`` call, we aren't running the test.
"""

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pytest
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression

import bayte as bt
from bayte.utils import make_regression, make_classification


@pytest.fixture
def regdf():
    """Test regression dataset."""
    return make_regression(
        dist="gamma",
        params=(1,),
        n_samples=100,
        n_features=10,
    )

@pytest.fixture
def clfdf():
    """Test classification dataset."""
    return make_classification()

def test_estimator_reg_fit(regdf):
    """Test a basic fit."""
    estimator = bt.BayesianTargetRegressor(
        base_estimator=SVR(kernel="linear"),
        encoder=bt.BayesianTargetEncoder(dist="gamma"),
        n_estimators=2
    )
    estimator.fit(*regdf, categorical_feature=[9,])

    assert hasattr(estimator, "estimators_")
    assert len(estimator.estimators_) == 2
    assert not np.array_equal(estimator.estimators_[0].coef_, estimator.estimators_[1].coef_)


def test_estimator_clf_fit(clfdf):
    """Test a basic fit with a classification task."""
    estimator = bt.BayesianTargetClassifier(
        base_estimator=LogisticRegression(),
        encoder=bt.BayesianTargetEncoder(dist="bernoulli"),
        n_estimators=2
    )
    estimator.fit(*clfdf)

    assert hasattr(estimator, "estimators_")
    assert len(estimator.estimators_) == 2
    assert not np.array_equal(estimator.estimators_[0].coef_, estimator.estimators_[1].coef_)

def test_estimator_fit_pandas(regdf):
    """Test a basic fit with a pandas DataFrame."""
    estimator = bt.BayesianTargetRegressor(
        base_estimator=SVR(kernel="linear"),
        encoder=bt.BayesianTargetEncoder(dist="gamma"),
        n_estimators=2
    )
    X, y = regdf
    X = pd.DataFrame(X)
    X[9] = X[9].astype("category")

    estimator.fit(X, y)

    assert_array_equal(
        estimator.categorical_, 
        np.array([False, False, False, False, False, False, False, False, False, True])
    )
    assert hasattr(estimator, "estimators_")
    assert len(estimator.estimators_) == 2
    assert not np.array_equal(
        estimator.estimators_[0].coef_,
        estimator.estimators_[1].coef_
    )


def test_estimator_reg_prefit(regdf):
    """Test a basic fit with a pre-fitted encoder."""
    X, y = regdf
    encoder = bt.BayesianTargetEncoder(dist="gamma")
    encoder.fit(X[:, [9]], y)

    estimator = bt.BayesianTargetRegressor(
        base_estimator=SVR(kernel="linear"),
        encoder=encoder,
        n_estimators=2,
    )
    estimator.fit(*regdf, categorical_feature=[9,])

    assert hasattr(estimator, "estimators_")
    assert len(estimator.estimators_) == 2
    assert not np.array_equal(
        estimator.estimators_[0].coef_,
        estimator.estimators_[1].coef_
    )


def test_estimator_reg_predict(regdf):
    """Test basic prediction with a regression dataset."""
    estimator = bt.BayesianTargetRegressor(
        base_estimator=SVR(kernel="linear"),
        encoder=bt.BayesianTargetEncoder(dist="gamma"),
        n_estimators=2
    )
    estimator.fit(*regdf, categorical_feature=[9,])

    y = estimator.predict(regdf[0])

    assert y.shape == (100,)


def test_estimator_clf_predict(clfdf):
    """Test basic prediction with a classification target."""
    estimator = bt.BayesianTargetClassifier(
        base_estimator=LogisticRegression(),
        encoder=bt.BayesianTargetEncoder(dist="bernoulli"),
        n_estimators=10
    )
    estimator.fit(*clfdf)

    y = estimator.predict(clfdf[0])
    yprob = estimator.predict_proba(clfdf[0])

    assert y.shape == (100,)
    assert_array_equal(np.unique(y), np.arange(2))
    assert yprob.shape == (100,2)
    assert ((yprob > 1) & (yprob < 0)).sum() == 0

    estimator.set_params(voting="soft")

    y = estimator.predict(clfdf[0])

    assert y.shape == (100,)
    assert_array_equal(np.unique(y), np.arange(2))

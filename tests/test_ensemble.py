"""Test the ensemble estimator.


NOTE: Since we can't control the ``categorical_feature`` input to ``fit`` in the
``sklearn.utils.estimator_checks.check_estimator`` call, we aren't running the test.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression

import bayte as bt
from bayte.utils import make_regression, make_classification


@pytest.fixture
def regdf():
    """Test regression dataset."""
    return make_regression("gamma", (1,))

@pytest.fixture
def clfdf():
    """Test classification dataset."""
    return make_classification()

def test_estimator_reg_fit(regdf):
    """Test a basic fit."""
    estimator = bt.BayesianTargetEstimator(
        base_estimator=SVR(kernel="linear"),
        encoder=bt.BayesianTargetEncoder(dist="gamma"),
        n_estimators=2
    )
    estimator.fit(*regdf)

    assert hasattr(estimator, "estimators_")
    assert len(estimator.estimators_) == 2
    assert estimator.estimators_[0].coef_ != estimator.estimators_[1].coef_


def test_estimator_clf_fit(clfdf):
    """Test a basic fit with a classification task."""
    estimator = bt.BayesianTargetEstimator(
        base_estimator=LogisticRegression(),
        encoder=bt.BayesianTargetEncoder(dist="bernoulli"),
        n_estimators=2
    )
    estimator.fit(*clfdf)

    assert hasattr(estimator, "estimators_")
    assert len(estimator.estimators_) == 2
    assert estimator.estimators_[0].coef_ != estimator.estimators_[1].coef_

def test_estimator_fit_pandas(regdf):
    """Test a basic fit with a pandas DataFrame."""
    estimator = bt.BayesianTargetEstimator(
        base_estimator=SVR(kernel="linear"),
        encoder=bt.BayesianTargetEncoder(dist="gamma"),
        n_estimators=2
    )
    X, y = regdf
    X = pd.DataFrame(X)
    X[0] = X[0].astype("category")

    estimator.fit(X, y)

    assert estimator.categorical_ == np.array([True])
    assert hasattr(estimator, "estimators_")
    assert len(estimator.estimators_) == 2
    assert estimator.estimators_[0].coef_ != estimator.estimators_[1].coef_


def test_estimator_reg_prefit(regdf):
    """Test a basic fit with a pre-fitted encoder."""
    encoder = bt.BayesianTargetEncoder(dist="gamma")
    encoder.fit(*regdf)

    estimator = bt.BayesianTargetEstimator(
        base_estimator=SVR(kernel="linear"),
        encoder=encoder,
        n_estimators=2,
    )
    estimator.fit(*regdf)

    assert hasattr(estimator, "estimators_")
    assert len(estimator.estimators_) == 2
    assert estimator.estimators_[0].coef_ != estimator.estimators_[1].coef_


def test_estimator_reg_predict(regdf):
    """Test basic prediction with a regression dataset."""
    estimator = bt.BayesianTargetEstimator(
        base_estimator=SVR(kernel="linear"),
        encoder=bt.BayesianTargetEncoder(dist="gamma"),
        n_estimators=2
    )
    estimator.fit(*regdf)

    y = estimator.predict(regdf[0])

    assert y.shape == (100,)


def test_estimator_clf_predict(clfdf):
    """Test basic prediction with a classification target."""
    estimator = bt.BayesianTargetEstimator(
        base_estimator=LogisticRegression(),
        encoder=bt.BayesianTargetEncoder(dist="bernoulli"),
        n_estimators=10
    )
    estimator.fit(*clfdf)

    y = estimator.predict(clfdf[0])
    yprob = estimator.predict_proba(clfdf[0])

    assert y.shape == (100,)
    assert yprob.shape == (100,2)
    assert ((yprob > 1) & (yprob < 0)).sum() == 0

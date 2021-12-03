"""Test the encoder."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest
import scipy.stats
from sklearn.utils.estimator_checks import check_estimator

from bayte.encoder import (
    BayesianTargetEncoder,
    _init_prior
)
from bayte.utils import make_regression

def test_encoder_validity():
    """Test the validity against the scikit-learn API."""
    check_estimator(BayesianTargetEncoder(dist="bernoulli"))


def test_init_prior_bernoulli():
    """Test initializing the prior distribution with a bernoulli likelihood."""
    y = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 1])
    out = _init_prior("bernoulli", y)

    assert out == (0.6, 0.4)


def test_init_prior_multinomial():
    """Test initializing the prior distribution with a multinomial likelihood."""
    y = np.array([0, 1, 2, 1, 0, 1, 1, 2, 2, 1])
    out = _init_prior("multinomial", y)

    assert out == (2, 5, 3)


def test_init_prior_exponential():
    """Test initializing the prior distribution with an exponential likelihood."""
    y = scipy.stats.expon(2).rvs(size=100)
    out = _init_prior("exponential", y)

    assert out == (101, np.sum(y))


def test_init_prior_gamma():
    """Test initializing the prior distribution with a gamma likelihood."""
    np.random.seed(42)
    y = scipy.stats.gamma(3).rvs(size=1000)
    out = _init_prior("gamma", y)

    assert np.abs(out[0]/1000 - 3) <= 1
    assert out[1] == 0
    assert out[2] == np.sum(y)


def test_init_prior_invgamma():
    """Test initializing the prior distribution with an inverse gamma likelihood."""
    np.random.seed(42)
    y = scipy.stats.invgamma(5).rvs(size=1000)
    out = _init_prior("invgamma", y)

    assert np.abs(out[0]/1000 - 5) <= 1
    assert out[1] == 0
    assert out[2] == np.sum(y)

def test_fit_invalid_dist():
    """Test raising an error with an invalid likelihood."""
    df = pd.DataFrame(
        {
            "x1": [0, 1, 2, 1, 0, 1, 2, 3, 3, 2],
            "y": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
        }
    )

    encoder = BayesianTargetEncoder(dist="fake")
    with pytest.raises(NotImplementedError):
        encoder.fit(df[["x1"]], df["y"])

def test_bernoulli_fit():
    """Test fitting the encoder with a binary classification task."""
    df = pd.DataFrame(
        {
            "x1": [0, 1, 2, 1, 0, 1, 2, 3, 3, 2],
            "y": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
        }
    )

    encoder = BayesianTargetEncoder(dist="bernoulli")
    encoder.fit(df[["x1"]], df["y"])

    assert hasattr(encoder, "prior_params_")
    assert encoder.prior_params_ == (0.5, 0.5)

    assert len(encoder.posterior_params_) == 1
    assert_array_equal(
        encoder.posterior_params_[0],
        np.array(
            [
                (1.5, 1.5, 0, 1),
                (1.5, 2.5, 0, 1),
                (1.5, 2.5, 0, 1),
                (2.5, 0.5, 0, 1)
            ]
        )
    )

    # Test parallel
    encoder.set_params(n_jobs=2)
    encoder.fit(df[["x1"]], df["y"])

    assert encoder.prior_params_ == (0.5, 0.5)


def test_multinomial_fit():
    """Test fitting the encoder with a multinomial classification task."""
    df = pd.DataFrame(
        {
            "x1": [0, 1, 2, 2, 2, 1, 1, 3, 3, 1, 2, 3, 3, 2, 0, 0, 0, 1, 0, 1],
            "y": [0, 1, 1, 1, 0, 2, 2, 1, 2, 0, 1, 0, 2, 1, 1, 0, 1, 1, 0, 2]
        }
    )

    encoder = BayesianTargetEncoder(dist="multinomial")
    encoder.fit(df[["x1"]], df["y"])

    assert encoder.prior_params_ == (6, 9, 5)
    assert len(encoder.posterior_params_) == 1
    assert_array_equal(
        encoder.posterior_params_[0],
        np.array(
            [
                (9, 11, 5),
                (7, 11, 8),
                (7, 13, 5),
                (7, 10, 7)
            ]
        )
    )

    # Test parallel
    encoder.set_params(n_jobs=2)
    encoder.fit(df[["x1"]], df["y"])

    assert encoder.prior_params_ == (6, 9, 5)


def test_multinomial_fit_missing_classes():
    """Test multinomial fit with missing target levels in categorical."""
    df = pd.DataFrame(
        {
            "x1": [0, 1, 0, 1, 1, 1, 0, 2, 1, 2],
            "y": [0, 0, 1, 1, 0, 2, 2, 1, 0, 2]
        }
    )

    encoder = BayesianTargetEncoder(dist="multinomial")
    encoder.fit(df[["x1"]], df["y"])

    assert encoder.prior_params_ == (4, 3, 3)
    assert len(encoder.posterior_params_) == 1
    assert_array_equal(
        encoder.posterior_params_[0],
        np.array(
            [
                (5, 4, 4),
                (7, 4, 4),
                (4, 4, 4)
            ]
        )
    )


def test_transform_bernoulli():
    """Test transforming with a bernoulli likelihood."""
    df = pd.DataFrame(
        {
            "x1": [0, 1, 2, 1, 0, 1, 2, 3, 3, 2],
            "y": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
        }
    )

    encoder = BayesianTargetEncoder(dist="bernoulli")
    encoder.fit(df[["x1"]], df["y"])

    out = encoder.transform(df[["x1"]])
    expected = np.array(
        [0.5, 0.375, 0.375, 0.375, 0.5, 0.375, 0.375, 0.833333, 0.833333, 0.375]
    )

    assert_allclose(out.ravel(), expected, rtol=1e-5)


    # Test parallel transform
    encoder.set_params(n_jobs=2)
    out = encoder.transform(df[["x1"]])

    assert_allclose(out.ravel(), expected, rtol=1e-5)

def test_transform_multinomial():
    """Test transforming with a multinomial likelihood."""
    df = pd.DataFrame(
        {
            "x1": [0, 1, 2, 2, 2, 1, 1, 3, 3, 1, 2, 3, 3, 2, 0, 0, 0, 1, 0, 1],
            "y": [0, 1, 1, 1, 0, 2, 2, 1, 2, 0, 1, 0, 2, 1, 1, 0, 1, 1, 0, 2]
        }
    )

    encoder = BayesianTargetEncoder(dist="multinomial")
    encoder.fit(df[["x1"]], df["y"])

    out = encoder.transform(df[["x1"]])
    expected = np.array(
        [
            [0.36, 0.44, 0.2],
            [0.26923, 0.423077, 0.307692],
            [0.28, 0.52, 0.2],
            [0.28, 0.52, 0.2],
            [0.28, 0.52, 0.2],
            [0.26923, 0.423077, 0.307692],
            [0.26923, 0.423077, 0.307692],
            [0.291667, 0.416667, 0.291667],
            [0.291667, 0.416667, 0.291667],
            [0.26923, 0.423077, 0.307692],
            [0.28, 0.52, 0.2],
            [0.291667, 0.416667, 0.291667],
            [0.291667, 0.416667, 0.291667],
            [0.28, 0.52, 0.2],
            [0.36, 0.44, 0.2],
            [0.36, 0.44, 0.2],
            [0.36, 0.44, 0.2],
            [0.26923, 0.423077, 0.307692],
            [0.36, 0.44, 0.2],
            [0.26923, 0.423077, 0.307692],
        ]
    )

    assert_allclose(out, expected, rtol=1e-5)

    # Test parallel transform
    encoder.set_params(n_jobs=2)
    out = encoder.transform(df[["x1"]])

    assert_allclose(out, expected, rtol=1e-5)


def test_transform_exponential():
    """Test transforming with an exponential likelihood."""
    X, y = make_regression("expon", (1,), n_samples=10000, n_levels=3)

    encoder = BayesianTargetEncoder(dist="exponential")
    encoder.fit(X, y)
    out = encoder.transform(X)

    assert len(encoder.posterior_params_[0]) == 3

    for index, params in enumerate(encoder.posterior_params_[0]):
        assert params[1] == 0
        assert params[2] == (np.sum(y)/(1 + np.sum(y) * np.sum(y[X[:, 0] == index + 1])))

        # Mean of posterior is params[0] * params[2]
        assert np.unique(out[X[:, 0] == index + 1]) == np.array([params[0] * params[2]])

    # Test parallel transform
    encoder.set_params(n_jobs=2)
    paraout = encoder.transform(X)

    assert_array_equal(out, paraout)


def test_transform_gamma():
    """Test transforming with a gamma likelihood."""
    X, y = make_regression("gamma", (1,), n_samples=10000, n_levels=4)

    encoder = BayesianTargetEncoder(dist="gamma")
    encoder.fit(X, y)
    out = encoder.transform(X)

    assert len(encoder.posterior_params_[0]) == 4

    for index, params in enumerate(encoder.posterior_params_[0]):
        assert np.unique(out[X[:, 0] == index + 1]) == np.array([params[0] * params[2]])

    # Test parallel transform
    encoder.set_params(n_jobs=2)
    paraout = encoder.transform(X)

    assert_array_equal(out, paraout)

"""Test the encoder."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import check_estimator

from bayte.encoder import (
    BayesianTargetEncoder,
    _init_prior
)

def test_encoder_validity():
    """Test the validity against the scikit-learn API."""
    check_estimator(BayesianTargetEncoder(dist="bernoulli"))

@pytest.mark.parametrize(
    "dist,target,params",
    [
        ("bernoulli", np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 1]), (0.6, 0.4)),
        ("multinomial", np.array([0, 1, 2, 1, 0, 1, 1, 2, 2, 1]), (2, 5, 3)),
    ]
)
def test_init_prior(dist, target, params):
    """Test initializing the prior distribution."""
    out = _init_prior(dist, target)

    assert out == params


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

    assert_allclose(
        out.ravel(),
        np.array(
            [0.5, 0.375, 0.375, 0.375, 0.5, 0.375, 0.375, 0.833333, 0.833333, 0.375]
        ),
        rtol=1e-5
    )


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

    assert_allclose(
        out,
        np.array(
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
        ),
        rtol=1e-5
    )

"""Test the encoder."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from bayesian_target.encoder import (
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

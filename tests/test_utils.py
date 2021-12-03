"""Test the data generation."""

import numpy as np
from numpy.testing import assert_array_equal
import pytest
import scipy.stats

from bayte.utils import make_categorical_regressor


@pytest.mark.parametrize(
    "dist,n_samples,n_levels",
    [
        ("norm", 1000, 5),
        ("norm", 1000, 10),
        ("norm", 1000, 20),
        ("norm", 1000, 50),
        ("norm", 10000, 100),
        ("expon", 1000, 5),
        ("expon", 1000, 10),
        ("expon", 1000, 20),
        ("expon", 1000, 50),
        ("expon", 10000, 100),
        ("gamma", 1000, 5),
        ("gamma", 1000, 10),
        ("gamma", 1000, 20),
        ("gamma", 1000, 50),
        ("gamma", 10000, 100),
    ]
)
def test_reg_gen(dist, n_samples, n_levels):
    """Test generating some data with a pre-specified distribution.
    
    Since there's randomness in sampling, we're testing to make sure we've
    created a categorical variable that's correlated with the target (i.e.
    we have an inherent ordinal categorical that we will treat as nominal).
    """
    X, y = make_categorical_regressor(
        dist=dist,
        params=(1,),
        n_samples=n_samples,
        n_levels=n_levels,
        random_state=42
    )

    assert_array_equal(np.unique(X[:, 0]), np.arange(1, n_levels + 1))

    avgs = []
    for l in range(1, n_levels + 1):
        # Fit the dist
        rv = getattr(scipy.stats, dist)
        curr_params = rv.fit(y[X[:, 0] == l])
        avgs.append(rv(*curr_params).mean())
    
    assert np.corrcoef(avgs) > 0.8

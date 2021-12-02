"""Test the data generation."""

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from bayte.utility import make_categorical_regressor


@pytest.mark.parametrize(
    "dist,n_levels", [("normal", 5), ("exponential", 10), ("gamma", 20)]
)
def test_reg_gen(dist, n_levels):
    """Test generating some data with a pre-specified distribution."""
    X, y = make_categorical_regressor(
        dist=dist,
        params=(1,),
        n_samples=10000,
        n_levels=n_levels,
    )

    assert_array_equal(np.unique(X[:, 0]), np.arange(n_levels))

    for l in range(1, n_levels):
        assert np.average(y[X[:, 0] == l]) > np.average(y[X[:, 0] == l - 1])

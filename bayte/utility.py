"""Utility functions for creating dummy data."""

from collections import deque
from typing import Tuple

import numpy as np
from sklearn.utils.validation import check_random_state


def make_categorical_regressor(
    dist: str,
    params: Tuple,
    n_samples: int = 100,
    n_levels: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a categorical column with a continuous target.

    Parameters
    ----------
    dist : str
        The likelihood for the target. Any distributions available through the numpy random
        state object (see `here <https://numpy.org/doc/stable/reference/random/legacy.html?highlight=randomstate#numpy.random.RandomState>`_)  # noqa: E501
        for availability.
    params
        Parameters for the likelihood
    n_samples : int, optional (default 100)
        The number of samples to generate.
    n_levels : int, optional (default 2)
        The number of levels for the categorical column.
    random_state : int
        The random state, optional

    Returns
    -------
    np.ndarray of shape (n_samples,)
        The categorical column.
    np.ndarray of shape (n_samples,)
        The target.
    """
    rng = check_random_state(random_state)
    y = getattr(rng, dist)(*params, size=n_samples)

    x = np.zeros(n_samples)
    # Get the quantiles
    classes_ = np.arange(n_levels)
    quantiles_to_get = np.linspace(0, 1, num=n_levels, endpoint=False)
    quantiles = np.quantile(y, quantiles_to_get[1:])

    # Create a probability vector and rotate it as we move through target
    # quantiles. This means one level in the categorical will be favoured
    # per quantile of the target.
    prob_vector = deque([0.75] + [0.25 / (n_levels - 1)] * (n_levels - 1))
    x[y < quantiles[0]] = np.random.choice(
        classes_, size=(y < quantiles[0]).sum(), p=prob_vector
    )
    for n in range(1, n_levels - 1):
        prob_vector.rotate()
        x[(y >= quantiles[n - 1]) & (y < quantiles[n])] = np.random.choice(
            classes_,
            size=((y >= quantiles[n - 1]) & (y < quantiles[n])).sum(),
            p=prob_vector,
        )
    prob_vector.rotate()
    x[y >= quantiles[-1]] = np.random.choice(
        classes_, size=(y >= quantiles[-1]).sum(), p=prob_vector
    )

    X = x.reshape((n_samples, 1))

    return X, y

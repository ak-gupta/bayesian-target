"""Utility functions for creating dummy data."""

from collections import deque
from typing import Optional, Tuple

import numpy as np
import scipy.stats


def make_categorical_regressor(
    dist: str,
    params: Tuple,
    n_samples: int = 100,
    n_levels: int = 2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a categorical column with a continuous target.

    Parameters
    ----------
    dist : str
        The likelihood for the target. Any distributions available through ``scipy.stats``
        can be used.
    params
        Parameters for the likelihood
    n_samples : int, optional (default 100)
        The number of samples to generate.
    n_levels : int, optional (default 2)
        The number of levels for the categorical column.
    random_state : int, optional (default None)
        The random state.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        The categorical column.
    np.ndarray of shape (n_samples,)
        The target.
    """
    y = getattr(scipy.stats, dist).rvs(*params, size=n_samples)

    catvar = np.zeros(n_samples)
    # Get the quantiles
    levels_ = np.arange(1, n_levels + 1)
    quantiles_to_get = np.linspace(0, 1, num=n_levels, endpoint=False)
    quantiles = np.quantile(y, quantiles_to_get[1:])

    # Create a probability vector and rotate it as we move through target
    # quantiles. This means one level in the categorical will be favoured
    # per quantile of the target.
    rv = scipy.stats.pareto(1)
    raw_ = np.linspace(rv.ppf(0.01), rv.ppf(0.99), num=n_levels)
    prob_vector = deque(rv.pdf(raw_) / rv.pdf(raw_).sum())
    catvar[y < quantiles[0]] = np.random.choice(
        levels_, size=(y < quantiles[0]).sum(), p=prob_vector
    )
    for n in range(1, n_levels - 1):
        prob_vector.rotate()
        catvar[(y >= quantiles[n - 1]) & (y < quantiles[n])] = np.random.choice(
            levels_,
            size=((y >= quantiles[n - 1]) & (y < quantiles[n])).sum(),
            p=prob_vector,
        )
    prob_vector.rotate()
    catvar[y >= quantiles[-1]] = np.random.choice(
        levels_, size=(y >= quantiles[-1]).sum(), p=prob_vector
    )

    X = catvar.reshape((n_samples, 1))

    return X, y


def make_categorical_classifier(
    n_samples: int = 100,
    n_levels: int = 2,
    n_classes: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a categorical column with a discrete target.
    
    Parameters
    ----------
    n_samples : int, optional (default 100)
        The number of samples to generate.
    n_levels : int, optional (default 2)
        The number of levels for the categorical column.
    n_classes: int, optional (default 2)
        The number of classes in the target variable.
    
    Returns
    -------
    np.ndarray of shape (n_samples,)
        The categorical column.
    np.ndarray of shape (n_samples,)
        The target.
    """
    y = np.random.choice(np.arange(n_classes), size=n_samples)

    # Create a probability vector and rotate it as we move through the
    # target classes. This means one level in the categorical will be favoured
    # per class in the target.
    catvar = np.zeros(n_samples)
    levels_ = np.arange(1, n_levels + 1)
    rv = scipy.stats.expon(1)
    raw_ = np.linspace(rv.ppf(0.01), rv.ppf(0.99), num=n_levels)
    prob_vector = deque(rv.pdf(raw_) / rv.pdf(raw_).sum())
    for n in range(1, n_classes + 1):
        catvar[y == n - 1] = np.random.choice(
            levels_, size=(y == n - 1).sum(), p=prob_vector
        )
        prob_vector.rotate()
    
    X = catvar.reshape((n_samples, 1))

    return X, y

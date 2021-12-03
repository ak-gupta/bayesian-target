"""Utility functions for creating dummy data."""

from collections import deque
from typing import Optional, Tuple

import numpy as np
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_random_state


def make_regression(
    dist: str,
    params: Tuple,
    n_samples: int = 100,
    n_features: int = 100,
    n_informative: int = 10,
    n_levels: int = 2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a categorical column with a continuous target.

    This is a limited implementation of the ``sklearn.datasets.make_regression`` function.
    Essentially, this function lets the user specify the distribution of the target as well
    as the number of levels in the final feature, which will be categorical and correlated
    with the target.

    Parameters
    ----------
    dist : str
        The likelihood for the target. Any distributions available through ``scipy.stats``
        can be used.
    params
        Parameters for the likelihood
    n_samples : int, optional (default 100)
        The number of samples to generate.
    n_features : int, optional (default 100)
        The number of features to generate. One of the generated features will be categorical.
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
    generator = check_random_state(random_state)
    # Randomly generate an input set
    X = generator.randn(n_samples, n_features)

    # Generate a ground truth model with n_informative features being non-zero
    ground_truth = np.zeros((n_features, 1))
    ground_truth[:n_informative, :] = 100 * generator.randn(n_informative, 1)

    y = np.dot(X, ground_truth)
    targetrv = getattr(scipy.stats, dist)(*params)
    noise = targetrv.rvs(size=n_samples)
    y += noise.reshape((n_samples, 1))
    # Shift the target based on the support of the underlying distribution
    support_ = list(targetrv.support())
    if np.sum([np.isfinite(val) for val in support_]) > 0:
        for idx in range(len(support_)):
            if np.isinf(support_[idx]):
                support_[idx] = np.max(y[:, 0])
            elif np.isneginf(support_[idx]):
                support_[idx] = np.min(y[:, 0])
        
        scaler = MinMaxScaler(feature_range=support_)
        y = scaler.fit_transform(y)

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
    catvar[y[:, 0] < quantiles[0]] = np.random.choice(
        levels_, size=(y < quantiles[0]).sum(), p=prob_vector
    )
    for n in range(1, n_levels - 1):
        prob_vector.rotate()
        catvar[(y[:, 0] >= quantiles[n - 1]) & (y[:, 0] < quantiles[n])] = np.random.choice(
            levels_,
            size=((y[:, 0] >= quantiles[n - 1]) & (y[:, 0] < quantiles[n])).sum(),
            p=prob_vector,
        )
    prob_vector.rotate()
    catvar[y[:, 0] >= quantiles[-1]] = np.random.choice(
        levels_, size=(y[:, 0] >= quantiles[-1]).sum(), p=prob_vector
    )

    X[:, n_features - 1] = catvar

    return X, y


def make_classification(
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

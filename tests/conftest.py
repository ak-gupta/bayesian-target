"""Fixtures for testing."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture
def toy_regression_dataset():
    """Toy regression dataset."""
    X, y = make_regression(n_samples=1000, n_features=10, n_informative=9)
    X[:, 9] = np.random.choice([0, 1, 2], size=1000)

    return X, y


@pytest.fixture
def toy_classification_dataset():
    """Toy classification dataset."""
    X, y = make_classification(n_samples=1000, n_features=10)
    X[:, 9] = np.random.choice([0, 1, 2], size=1000)

    return X, y

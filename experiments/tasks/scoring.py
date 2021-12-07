"""Task for model scoring."""

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from prefect import task
from sklearn.base import is_classifier
from sklearn.metrics._scorer import check_scoring
from sklearn.model_selection._split import check_cv
from sklearn.utils.metaestimators import _safe_split

SCORER = {"regression": "neg_root_mean_squared_error", "classification": "roc_auc"}


@task(name="Split data")
def split_data(data: pd.DataFrame, metadata: Dict, estimator) -> List[Tuple]:
    """Split the dataset into 5 folds for cross-validation.

    Parameters
    ----------
    data : pd.DataFrame
        The training data.
    metadata : dict
        The metadata dictionary.
    estimator : object
        The estimator object for this experiment.

    Returns
    -------
    List
        A list of tuples, each index indicating a train-test split for scoring.
    """
    cv = check_cv(5, data[metadata["target"]], classifier=is_classifier(estimator))

    return list(
        cv.split(
            data[metadata["numeric"] + metadata["nominal"]],
            data[metadata["target"]],
            None,
        )
    )


@task(name="Cross-validated scoring")
def fit_and_score_model(
    data: pd.DataFrame, metadata: Dict, estimator, splits: Tuple, encoder=None
) -> Tuple[float, float]:
    """Fit and score a model fold.

    Parameters
    ----------
    data : pd.DataFrame
        The training data.
    metadata : dict
        The metadata dictionary
    estimator : object
        The estimator object for this experiment
    splits : tuple
        The train/test split for this iteration
    encoder : object, optional (default None)
        The categorical encoder, if required.

    Returns
    -------
    float
        Fold score.
    float
        The fitting time, in seconds.
    """
    scorers = check_scoring(estimator, SCORER[metadata["dataset_type"]])
    X_train, y_train = _safe_split(
        estimator,
        data[metadata["numeric"] + metadata["nominal"]],
        data[metadata["target"]],
        splits[0],
    )
    X_test, y_test = _safe_split(
        estimator,
        data[metadata["numeric"] + metadata["nominal"]],
        data[metadata["target"]],
        splits[1],
    )
    start_time = time.time()
    if encoder is not None:
        categorical_ = np.zeros(data.shape[0], dtype=bool)
        for idx, col in enumerate(data.columns):
            if col in metadata["nominal"]:
                categorical_[idx] = True

        X_train_encoded = encoder.fit_transform(X_train[:, categorical_], y_train)
        X_train = np.hstack((X_train[:, ~categorical_], X_train_encoded))

        X_test_encoded = encoder.transform(X_test[:, categorical_])
        X_test = np.hstack((X_test[:, ~categorical_], X_test_encoded))

    estimator.fit(X_train, y_train)
    fit_time = time.time() - start_time

    return scorers(estimator, X_test, y_test), fit_time

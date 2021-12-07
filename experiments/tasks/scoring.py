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
    columns = metadata["numeric"] + metadata["nominal"]
    X_train, y_train = _safe_split(
        estimator, data[columns].to_numpy(), data[metadata["target"]].to_numpy(), splits[0],
    )
    X_test, y_test = _safe_split(
        estimator,
        data[columns].to_numpy(),
        data[metadata["target"]].to_numpy(),
        splits[1],
    )
    start_time = time.time()
    categorical_ = np.zeros(X_train.shape[1], dtype=bool)
    for idx, col in enumerate(columns):
        if col in metadata["nominal"]:
            categorical_[idx] = True

    if encoder is not None:
        X_train_encoded = encoder.fit_transform(X_train[:, categorical_], y_train)
        X_train = np.hstack((X_train[:, ~categorical_], X_train_encoded))

        X_test_encoded = encoder.transform(X_test[:, categorical_])
        X_test = np.hstack((X_test[:, ~categorical_], X_test_encoded))
    else:
        X_train = X_train[:, ~categorical_]
        X_test = X_test[:, ~categorical_]

    estimator.fit(X_train, y_train)
    fit_time = time.time() - start_time

    return scorers(estimator, X_test, y_test), fit_time


@task(name="Average score")
def avg_score(scoring_out: List[Tuple]) -> float:
    """Get the average score.
    
    Parameters
    ----------
    scoring_out : List
        The test score and training time from each fold.
    
    Returns
    -------
    float
        The average score.
    """
    return np.average([out[0] for out in scoring_out])


@task(name="Average fit time")
def avg_fit_time(scoring_out: List[Tuple]) -> float:
    """Get the average fit time.
    
    Parameters
    ----------
    scoring_out : list
        The test score and training time for each fold.
    
    Returns
    -------
    float
        The average fit time.
    """
    return np.average([out[1] for out in scoring_out])

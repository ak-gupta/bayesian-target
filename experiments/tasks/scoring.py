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

from bayte import BayesianTargetEncoder, BayesianTargetRegressor

SCORER = {"regression": "neg_root_mean_squared_error", "classification": "roc_auc"}


@task(name="Split data")
def split_data(data: pd.DataFrame, metadata: Dict, estimator) -> List[Tuple]:
    """Split the dataset into 10 folds for cross-validation.

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
    cv = check_cv(10, data[metadata["target"]], classifier=is_classifier(estimator))

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
        estimator,
        data[columns].to_numpy(),
        data[metadata["target"]].to_numpy(),
        splits[0],
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


@task(name="Cross-validated ensemble scoring")
def fit_and_score_ensemble_model(
    data: pd.DataFrame,
    metadata: Dict,
    estimator,
    splits: Tuple,
    encoder: BayesianTargetEncoder,
    n_estimators: int,
) -> Tuple[float, float]:
    """Fit and score an ensemble model with BTE.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset.
    metadata : dict
        Themetadata dictionary.
    estimator : object
        The scikit-learn compatible estimator object for the experiment.
    splits : tuple
        The train/test split for this fold.
    encoder : BayesianTargetEncoder
        The bayesian target encoder instance.
    n_estimators : int
        The number of samples to draw.

    Returns
    -------
    float
        The fold score.
    float
        The fit time, in seconds.
    """
    scorers = check_scoring(estimator, SCORER[metadata["dataset_type"]])
    columns = metadata["numeric"] + metadata["nominal"]
    X_train, y_train = _safe_split(
        estimator,
        data[columns].to_numpy(),
        data[metadata["target"]].to_numpy(),
        splits[0],
    )
    X_test, y_test = _safe_split(
        estimator,
        data[columns].to_numpy(),
        data[metadata["target"]].to_numpy(),
        splits[1],
    )
    start_time = time.time()

    if metadata["dataset_type"] == "regression":
        ensemble = BayesianTargetRegressor(
            base_estimator=estimator,
            encoder=encoder,
            n_estimators=n_estimators
        )
    else:
        raise NotImplementedError("Not implemented yet.")

    ensemble.fit(
        X_train,
        y_train,
        categorical_feature=[
            idx for idx, col in enumerate(columns) if col in metadata["nominal"]
        ],
    )
    fit_time = time.time() - start_time

    return scorers(ensemble, X_test, y_test), fit_time


@task(name="Average score")
def final_scores(scoring_out: List[Tuple]) -> List[float]:
    """Get the fold scores.

    Parameters
    ----------
    scoring_out : List
        The test score and training time from each fold.

    Returns
    -------
    list
        The fold scores
    """
    return [out[0] for out in scoring_out]


@task(name="Average fit time")
def final_fit_times(scoring_out: List[Tuple]) -> List[float]:
    """Get the fit times.

    Parameters
    ----------
    scoring_out : list
        The test score and training time for each fold.

    Returns
    -------
    list
        The fold scores.
    """
    return [out[1] for out in scoring_out]

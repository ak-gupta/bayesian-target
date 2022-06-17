"""Initialize the model and encoder."""

from typing import Dict

import pandas as pd
from prefect import task
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from xgboost import XGBClassifier, XGBRegressor

from bayte import BayesianTargetRegressor, BayesianTargetClassifier

@task(name="Check if ensemble")
def check_ensemble(n_estimators: int) -> bool:
    """Check if the current experiment uses ensemble encoding.

    Parameters
    ----------
    n_estimators : int
        The number of estimators. If 0, the experiment does not use sampling.

    Returns
    -------
    bool
        Evaluates to ``True`` if the flow should use ensemble.
    """
    return n_estimators > 0


@task(name="Initialize model")
def init_model(algorithm: str, metadata: Dict, seed: int = 42):
    """Initialize a model object.

    Parameters
    ----------
    algorithm : {"linear", "xgboost", "lightgbm"}
        The modelling package to use.
    metadata : dict
        The metadata configuration for the dataset.
    seed : int, optional (default 42)
        Random seed

    Returns
    -------
    object
        The initialized model.
    """
    if metadata["dataset_type"] == "regression":
        if algorithm == "xgboost":
            return XGBRegressor(random_state=seed)
        elif algorithm == "lightgbm":
            return LGBMRegressor(random_state=seed)
        elif algorithm == "gbm":
            return GradientBoostingRegressor(random_state=seed)
        else:
            raise NotImplementedError(f"{algorithm} is not a valid algorithm type.")
    elif metadata["dataset_type"] == "classification":
        if algorithm == "xgboost":
            return XGBClassifier(
                random_state=seed, use_label_encoder=False, eval_metric="logloss"
            )
        elif algorithm == "lightgbm":
            return LGBMClassifier(random_state=seed)
        elif algorithm == "gbm":
            return GradientBoostingClassifier(random_state=seed)
        else:
            raise NotImplementedError(f"{algorithm} is not a valid algorithm type.")


@task(name="Fit model")
def train(
    data: pd.DataFrame,
    metadata: Dict,
    estimator,
    encoder=None,
    n_estimators: int = 0,
    seed: int = 42
):
    """Fit the estimator.

    Parameters
    ----------
    data : pd.DataFrame
        The training data.
    metadata : Dict
        The metadata dictionary.
    estimator : object
        The scikit-learn compatible estimator object.
    encoder : object, optional (default None)
        A :py:class:`bayte.BayesianTargetEncoder` instance for ensemble modelling.
    n_estimators : int, optional (default 0)
        The number of samples to draw. If 0, no ensemble estimator will be used.
    seed : int, optional (default 42)
        The random state.

    Returns
    -------
    object
        Fitted estimator.
    """
    features = [col for col in data.columns if col != metadata["target"]]
    if n_estimators > 0:
        if metadata["dataset_type"] == "regression":
            estimator_ = BayesianTargetRegressor(
                base_estimator=estimator,
                encoder=encoder,
                n_estimators=n_estimators,
                random_state=seed
            )
        else:
            estimator_ = BayesianTargetClassifier(
                base_estimator=estimator,
                encoder=encoder,
                n_estimators=n_estimators,
                random_state=seed
            )

        estimator_.fit(
            data[features],
            data[metadata["target"]],
            categorical_feature=metadata["nominal"]
        )
    else:
        estimator_ = estimator.fit(data[features], data[metadata["target"]])

    return estimator_

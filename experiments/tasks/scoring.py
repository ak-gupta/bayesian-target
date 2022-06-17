"""Task for model scoring."""

from typing import Dict

import pandas as pd
from prefect import task
from sklearn.metrics._scorer import check_scoring

SCORER = {"regression": "neg_root_mean_squared_error", "classification": "roc_auc"}


@task(name="Score model")
def score_model(data: pd.DataFrame, metadata: Dict, estimator) -> float:
    """Score a model.

    Parameters
    ----------
    data : pd.DataFrame
        The encoded test data.
    metadata : dict
        The metadata dictionary
    estimator : object
        The estimator object for this experiment
    encoder : object, optional (default None)
        The categorical encoder, if required.

    Returns
    -------
    float
        Test score.
    """
    scorers = check_scoring(estimator, SCORER[metadata["dataset_type"]])
    features = [col for col in data.columns if col != metadata["target"]]

    return scorers(estimator, data[features], data[metadata["target"]])

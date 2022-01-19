"""Initialize the model and encoder."""

from typing import Dict

from category_encoders import (
    CountEncoder,
    GLMMEncoder,
    JamesSteinEncoder,
    TargetEncoder,
)
from prefect import task
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier, XGBRegressor

from bayte import BayesianTargetEncoder


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
            return XGBClassifier(random_state=seed)
        elif algorithm == "lightgbm":
            return LGBMClassifier(random_state=seed)
        elif algorithm == "gbm":
            return GradientBoostingClassifier(random_state=seed)
        else:
            raise NotImplementedError(f"{algorithm} is not a valid algorithm type.")


@task(name="Initialize encoder")
def init_encoder(algorithm: str, metadata: Dict):
    """Initialize a categorical encoder.

    Parameters
    ----------
    algorithm : {"frequency", "gllm", "james-stein", "one-hot", "integer", "target", "bayes"}
        The type of categorical encoder.
    metadata : dict
        The metadata configuration for the dataset.

    Returns
    -------
    object
        The initialized encoder.
    """
    if algorithm == "frequency":
        return CountEncoder()
    elif algorithm == "gllm":
        return GLMMEncoder()
    elif algorithm == "james-stein":
        return JamesSteinEncoder()
    elif algorithm == "one-hot":
        return OneHotEncoder()
    elif algorithm == "integer":
        return OrdinalEncoder()
    elif algorithm == "target":
        return TargetEncoder()
    elif algorithm == "bayes":
        return BayesianTargetEncoder(dist=metadata["dist"])

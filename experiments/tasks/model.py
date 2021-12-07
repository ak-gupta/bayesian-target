"""Initialize the model and encoder."""

from typing import Dict

from category_encoders import CountEncoder, GLLMEncoder, JamesSteinEncoder, TargetEncoder
from prefect import task
from lightgbm import LGBClassifier, LGBRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier, XGBRegressor

from bayte import BayesianTargetEncoder


@task(name="Initialize model")
def init_model(algorithm: str, metadata: Dict):
    """Initialize a model object.
    
    Parameters
    ----------
    algorithm : {"linear", "xgboost", "lightgbm"}
        The modelling package to use.
    metadata : dict
        The metadata configuration for the dataset.
    
    Returns
    -------
    object
        The initialized model.
    """
    if metadata["dataset_type"] == "regression":
        if algorithm == "linear":
            return LinearRegression()
        elif algorithm == "xgboost":
            return XGBRegressor()
        elif algorithm == "lightgbm":
            return LGBRegressor()
        else:
            raise NotImplementedError(f"{algorithm} is not a valid algorithm type.")
    elif metadata["dataset_type"] == "classification":
        if algorithm == "linear":
            return LogisticRegression()
        elif algorithm == "xgboost":
            return XGBClassifier()
        elif algorithm == "lightgbm":
            return LGBClassifier()
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
        return GLLMEncoder()
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

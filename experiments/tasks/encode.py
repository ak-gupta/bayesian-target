"""Encode the categorical values."""

from typing import Dict

from bayte import BayesianTargetEncoder
from category_encoders import (
    CountEncoder,
    GLMMEncoder,
    JamesSteinEncoder,
    TargetEncoder,
)
import numpy as np
import pandas as pd
import prefect
from prefect import task
from sklearn.preprocessing import OrdinalEncoder

@task(name="Check supervised")
def check_supervised(algorithm: str) -> bool:
    """Check if the encoder is supervised.

    Parameters
    ----------
    algorithm : {"frequency", "glmm", "james-stein", "integer", "target", "bayes"}
        The encoder.

    Returns
    -------
    bool
        True if the algorithm is supervised.
    """
    return bool(algorithm in ["glmm", "james-stein", "target", "bayes"])


@task(name="Initialize encoder")
def init_encoder(algorithm: str, metadata: Dict):
    """Initialize a categorical encoder.

    Parameters
    ----------
    algorithm : {"frequency", "glmm", "james-stein", "integer", "target", "bayes"}
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
    elif algorithm == "glmm":
        return GLMMEncoder()
    elif algorithm == "james-stein":
        return JamesSteinEncoder()
    elif algorithm == "integer":
        return OrdinalEncoder()
    elif algorithm == "target":
        return TargetEncoder()
    elif algorithm == "bayes":
        return BayesianTargetEncoder(dist=metadata["dist"], chunksize=300)


@task(name="Fit encoder")
def fit_encoder(
    data: pd.DataFrame,
    metadata: Dict,
    encoder,
    estimator=None,
    marginal: bool = False,
    residual: bool = False
):
    """Fit the encoder.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset. For unsupervised encoding algorithms, this will be
        the entire dataset to help identify all possible categorical values.
        For supervised methods, this will be the training set.
    metadata : Dict
        Dataset metadata.
    encoder : object
        The unfitted encoder.
    estimator : object, optional (default None)
        The optional estimator to be used for residual or marginal encoding.
    marginal : bool, optional (default False)
        Whether or not to encode categorical variables using the output of a model
        trained on non-categorical data. Only one of ``marginal`` or ``residual``
        can be set to ``True`` at once.
    residual : bool, optional (default False)
        Whether or not to encode categorical variables using the residuals from a
        model trained on non-categorical data. Only one of ``marginal`` or ``residual``
        can be set to ``True`` at once.

    Returns
    -------
    object
        The fitted encoder to be used downstream for transformation.
    """
    if marginal or residual:
        model = estimator.fit(
            data[metadata["numeric"]].to_numpy(), data[metadata["target"]].to_numpy()
        )
        y_marginal = model.predict(data[metadata["numeric"]].to_numpy())
        if marginal:
            return encoder.fit(data[metadata["nominal"]].to_numpy(), y_marginal)
        elif residual:
            y_residual = data[metadata["target"]] - y_marginal

            return encoder.fit(data[metadata["nominal"]].to_numpy(), y_residual)
    else:
        return encoder.fit(
            data[metadata["nominal"]].to_numpy(), data[metadata["target"]].to_numpy()
        )


@task(name="Encode data")
def transform(data: pd.DataFrame, encoder, metadata: Dict) -> pd.DataFrame:
    """Encode the data.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset.
    encoder : object
        Fitted encoder.
    metadata : Dict
        Dataset metadata.

    Returns
    -------
    pd.DataFrame
        The transformed dataset.
    """
    logger = prefect.context.get("logger")
    logger.info("Running transform...")
    encoder.set_params(sample=False)
    X_encoded = encoder.transform(data[metadata["nominal"]].to_numpy())
    logger.info("Data successfully transformed...")
    transformed = pd.DataFrame(
        np.hstack((data[metadata["numeric"]].to_numpy(), X_encoded))
    )
    transformed[metadata["target"]] = data[metadata["target"]]

    return transformed

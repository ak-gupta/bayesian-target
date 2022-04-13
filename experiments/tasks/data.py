"""Basic data parsing."""

from typing import Dict, Tuple

import pandas as pd
from prefect import task
from sklearn.model_selection import train_test_split


@task(name="Drop nulls")
def drop_nulls(data: pd.DataFrame) -> pd.DataFrame:
    """Drop null rows.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset.
    metadata : Dict
        The dataset metadata.

    Returns
    -------
    pd.DataFrame
        The updated dataframe.
    """
    return data.dropna()


@task(name="Split data", nout=2)
def split(
    data: pd.DataFrame, metadata: Dict, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into train and test splits.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset.
    metadata : Dict
        The dataset metadata.
    seed : int, optional (default 42)
        Random seed.

    Returns
    -------
    pd.DataFrame
        Training set.
    pd.DataFrame
        Test set.
    """
    return train_test_split(
        data[metadata["numeric"] + metadata["nominal"] + [metadata["target"]]],
        test_size=0.2,
        random_state=seed
    )

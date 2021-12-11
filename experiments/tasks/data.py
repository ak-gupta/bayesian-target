"""Basic data parsing.

Most of the regression datasets use an inverse gamma or gamma distribution.
Since these distributions are limited to a strictly positive domain, we will
offset the target variable by the minimum observed value and a buffer of 1e-5.
"""

from typing import Dict

import numpy as np
import pandas as pd
from prefect import task


@task(name="Create positive target")
def drop_nulls(data: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
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

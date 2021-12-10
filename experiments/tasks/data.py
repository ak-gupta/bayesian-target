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
def positive_target(data: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
    """Create a positive target and drop nulls.

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
    new = data.copy().dropna()
    if metadata["dist"] in ("exponential", "gamma", "invgamma"):
        if np.min(data[metadata["target"]]) < 0:
            new[metadata["target"]] += np.min(new[metadata["target"]])
            new[metadata["target"]] += 1e-5

    return new

"""Parsing lazyscribe project data."""

from typing import List

import pandas as pd
from prefect import task


@task(name="Create plotting dataframe")
def project_to_df(data: List) -> pd.DataFrame:
    """Convert experimental data to a dataframe.

    Parameters
    ----------
    data : List
        The project data.

    Returns
    -------
    pd.DataFrame
        The dataframe.
    """
    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    return df

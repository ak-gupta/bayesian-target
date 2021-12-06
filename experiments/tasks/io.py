"""File I/O."""

import json
from typing import Dict

import pandas as pd
from prefect import task

from .. import DATA_DIR, METADATA_DIR


@task(name="Read Metadata")
def read_metadata(dataset: str) -> Dict:
    """Read dataset metadata.
    
    Parameters
    ----------
    dataset : str
        The name of the dataset
    
    Returns
    -------
    Dict
        The metadata configuration.
    """
    fpath = METADATA_DIR / f"{dataset}.json"
    if not fpath.is_file():
        raise FileNotFoundError(f"No metadata for {dataset} available.")
    with open(fpath) as infile:
        return json.load(infile)


@task(name="Read dataset")
def read_data(metadata: Dict) -> pd.DataFrame:
    """Read the dataset file.
    
    Parameters
    ----------
    metadata : Dict
        The output from ``read_metadata``.
    
    Returns
    -------
    pd.DataFrame
        The output dataset.
    """
    return pd.read_csv(DATA_DIR / metadata["dataset_type"] / metadata["local_file"])

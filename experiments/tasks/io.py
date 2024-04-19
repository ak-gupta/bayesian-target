"""File I/O."""

import json
from typing import Dict

import pandas as pd
from prefect import task
from scipy.io.arff import loadarff

from experiments import DATA_DIR, METADATA_DIR


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
    fpath = DATA_DIR / metadata["dataset_type"] / metadata["local_file"]
    if fpath.suffix == ".arff":
        raw = loadarff(fpath)
        data = pd.DataFrame(raw[0])
    elif fpath.suffix == ".csv":
        data = pd.read_csv(fpath)
    else:
        raise NotImplementedError(f"File type `{fpath.suffix}` not supported.")

    if (
        metadata["dataset_type"] == "classification"
        and pd.api.types.infer_dtype(data[metadata["target"]]) == "bytes"
    ):
        data[metadata["target"]] = data[metadata["target"]].astype(int)

    return data

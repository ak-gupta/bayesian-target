"""Parsing rubicon project data."""

import pandas as pd
from prefect import task
from rubicon_ml.client.project import Project


@task(name="Parse base performance data")
def parse_base_performance(project: Project) -> pd.DataFrame:
    """Parse base performance data into a DataFrame.

    Parameters
    ----------
    project : Project
        The rubicon project.

    Returns
    -------
    pd.DataFrame
        A dataframe with experiment data.
    """
    dflist = []
    for experiment in project.experiments():
        data = {"fit-time": [], "score": []}
        for parameter in experiment.parameters():
            if parameter.name in ("dataset", "algorithm"):
                data[parameter.name] = parameter.value
        for metric in experiment.metrics():
            if metric.name.startswith("score"):
                data["score"].append(metric.value)
            elif metric.name.startswith("fit-time"):
                data["fit-time"].append(metric.value)
        dflist.append(pd.DataFrame(data))

    return pd.concat(dflist, ignore_index=True)


@task(name="Parse sampling performance data")
def parse_sampling_performance(project: Project) -> pd.DataFrame:
    """Parse sampling performance data into a DataFrame.

    Parameters
    ----------
    project : Project
        The rubicon project.

    Returns
    -------
    pd.DataFrame
        A dataframe with experiment data.
    """
    dflist = []
    for experiment in project.experiments():
        data = {"fit-time": [], "score": []}
        for parameter in experiment.parameters():
            if parameter.name in ("dataset", "algorithm", "n_estimators"):
                data[parameter.name] = parameter.value
        for metric in experiment.metrics():
            if metric.name.startswith("score"):
                data["score"].append(metric.value)
            elif metric.name.startswith("fit-time"):
                data["fit-time"].append(metric.value)
        dflist.append(pd.DataFrame(data))

    return pd.concat(dflist, ignore_index=True)

"""Target distribution visualization."""

from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from prefect import task
import seaborn as sns

from bayte.plots import visualize_target_dist

from .. import OUTPUT_DIR


@task(name="Create performance data")
def create_plot_df(base: pd.DataFrame, exp: pd.DataFrame) -> pd.DataFrame:
    """Create a plotting dataframe for performance.
    
    Parameters
    ----------
    base : pd.DataFrame
        The base performance data.
    exp : pd.DataFrame
        The experiment performance data.
    
    Returns
    -------
    pd.DataFrame
        Aggregated performance data.
    """
    grouped = base.groupby("dataset")
    for name, group in grouped:
        agg = group.groupby("algorithm")[["fit-time", "score"]].mean().reset_index()
        agg.rename(columns={"fit-time": "fit-time-base", "score": "score-base"}, inplace=True)
        agg["dataset"] = name
        tmp = exp[["dataset", "algorithm"]].merge(
            agg, how="left", left_on=("algorithm", "dataset"), right_on=("algorithm", "dataset")
        )

        exp.loc[exp["dataset"] == name, "score-base"] = tmp["score-base"]
        exp.loc[exp["dataset"] == name, "fit-time-base"] = tmp["fit-time-base"]
    
    exp["score-change"] = (exp["score"] - exp["score-base"])/exp["score-base"]
    exp["fit-time-change"] = (exp["fit-time"] - exp["fit-time-base"])/exp["fit-time-base"]

    return exp


@task(name="Visualize target distribution")
def render_dist_plot(data: pd.DataFrame, metadata: Dict):
    """Render the target distribution plot to a file.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.
    metadata : dict
        The metadata dictionary.
    """
    with sns.axes_style("dark"):
        fig = visualize_target_dist(
            data[metadata["target"]], metadata["candidate_dists"]
        )
        fig.suptitle(f"{metadata['dataset_name']} target density")

    fig.savefig(OUTPUT_DIR / f"{metadata['dataset_name']}.png")


@task(name="Visualize sampling performance")
def render_sample_perf_plot(data: pd.DataFrame):
    """Render performance data for the sampling experiment.

    Renders the plots to ``{dataset}-sampling.png``.

    Parameters
    ----------
    data : pd.DataFrame
        The output from ``create_plot_df``.
    """
    grouped = data.groupby("algorithm")

    for name, group in grouped:
        with sns.axes_style("dark"):
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.pointplot(
                x="n_estimators",
                y="score-change",
                hue="dataset",
                ci=None,
                data=group,
                palette="flare",
                dodge=True,
                ax=ax
            ).set(
                title=f"Effect of number of samples on performance for {name}",
                xlabel="Number of estimators (0 indicates no sampling)",
                ylabel="Average change in performance metric vs base"
            )
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            fig.tight_layout()
        
        fig.savefig(OUTPUT_DIR / f"{name}-sampling.png")

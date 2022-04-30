"""Target distribution visualization."""

from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from prefect import task
import seaborn as sns

from bayte.plots import visualize_target_dist

from .. import OUTPUT_DIR


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
    def _single_plot(name: str, data: pd.DataFrame):
        """Create a single plot."""
        with sns.axes_style("dark"):
            # Get the non-sampling performance
            non_sample = data.loc[
                data[("parameters", "n_estimators")] == 0, ("metrics", "score")
            ].mean()
            data["score-change"] = (data[("metrics", "score")] - non_sample) / abs(non_sample)
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.pointplot(
                x=("parameters", "n_estimators"),
                y="score-change",
                hue=("parameters", "model"),
                data=data[data[("parameters", "n_estimators")] > 0],
                palette="flare",
                dodge=True,
                ci="sd",
                ax=ax
            ).set(
                title=f"Effect of number of samples on performance for {name}",
                xlabel="Number of estimators",
                ylabel="Score change vs. no sampling (higher is better)"
            )
            plt.legend(title="Model")
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            fig.tight_layout()

        return fig


    grouped = data.groupby(("parameters", "dataset"))
    for name, group in grouped:
        fig = _single_plot(name, group)
        fig.savefig(OUTPUT_DIR / f"{name}-sampling.png")

    # Create overall plot
    fig = _single_plot(name="all datasets", data=data)
    fig.savefig(OUTPUT_DIR / "all-datasets-sampling.png")


@task(name="Visualize comparison performance")
def render_comparison_perf_plot(data: pd.DataFrame):
    """Render performance plots for the comparison experiment."""
    def _single_plot(name: str, data: pd.DataFrame):
        with sns.axes_style("dark"):
            # Get the baseline performance
            mask = (
                (data[("parameters", "n_estimators")] == 0)
                & (data[("parameters", "marginal")] == False)
                & (data[("parameters", "residual")] == False)
                & (data[("parameters", "encoder")] == "bayes")
            )
            baseline = data.loc[mask, ("metrics", "score")].mean()
            data["score-change"] = (data[("metrics", "score")] - baseline) / abs(baseline)
            data.loc[
                data[("parameters", "marginal")] == True, ("parameters", "encoder")
            ] += " (m)"
            data.loc[
                data[("parameters", "residual")] == True, ("parameters", "encoder")
            ] += " (r)"
            data.loc[
                (
                    (data[("parameters", "n_estimators")] > 0)
                    & (data[("parameters", "encoder")].str.startswith("bayes"))
                ),
                ("parameters", "encoder")
            ] += " (" + data[("parameters", "n_estimators")].astype(str) + ")"

            fig, ax = plt.subplots(figsize=(12, 8))
            #sns.pointplot(
            sns.violinplot(
                x=("parameters", "encoder"),
                y="score-change",
                hue=("parameters", "model"),
                data=data[~mask],
                palette="flare",
                inner="points",
                #dodge=True,
                #ci="sd",
                ax=ax
            ).set(
                title=f"Effect of encoder on performance for {name}",
                xlabel="Encoder (number of samples)",
                ylabel="Score change vs. standard bayes (higher is better)"
            )
            plt.legend(title="Model")
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            fig.tight_layout()

        return fig

    grouped = data.groupby(("parameters", "dataset"))
    for name, group in grouped:
        fig = _single_plot(name, group)
        fig.savefig(OUTPUT_DIR / f"{name}-comparison.png")

    # Create overall plot
    fig = _single_plot(name="all datasets", data=data)
    fig.savefig(OUTPUT_DIR / "all-datasets-comparison.png")

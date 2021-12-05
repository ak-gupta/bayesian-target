"""Helpful visualizations for target encoding."""

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns


def visualize_target_dist(y: np.ndarray) -> Figure:
    """Produce a histogram for the target variable with traces.

    This function will create a histogram of the target and
    layer traces for any compatible distribution that is also
    available for Bayesian Target Encoding.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target values.

    Returns
    -------
    Figure
        The figure object to persist or display
    """
    candidates_ = {
        "expon": "exponential",
        "gamma": "gamma",
        "invgamma": "invgamma",
        "norm": "normal",
    }
    dflist = []
    for dist, label in candidates_.items():
        params = getattr(scipy.stats, dist).fit(y)
        rv = getattr(scipy.stats, dist)(*params)
        x = np.linspace(rv.ppf(0.01), rv.ppf(0.99))
        dflist.append(pd.DataFrame({"x": x, "y": rv.pdf(x), "dist": label}))

    tracedf = pd.concat(dflist, ignore_index=True)

    plt.ioff()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(y, stat="density", ax=ax)
    sns.lineplot(x="x", y="y", hue="dist", data=tracedf, palette="flare", ax=ax)

    fig.tight_layout()

    return fig

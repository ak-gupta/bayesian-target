"""Helpful visualizations for target encoding."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from matplotlib.figure import Figure

DIST_MAPPING: Dict = {
    "exponential": "expon",
    "gamma": "gamma",
    "invgamma": "invgamma",
    "normal": "norm",
}

DEFAULT_CANDIDATES: List[str] = ["exponential", "gamma", "invgamma", "normal"]


def visualize_target_dist(
    y: np.ndarray,
    candidates: Optional[List[str]] = None,
) -> Figure:
    """Produce a histogram for the target variable with traces.

    This function will create a histogram of the target and
    layer traces for any compatible distribution that is also
    available for Bayesian Target Encoding.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target values.
    candidates : list, optional (default None)
        The candidate likelihoods to consider. By default, the following distributions
        will be visualized:

        * ``exponential``,
        * ``gamma``,
        * ``invgamma``, and
        * ``normal``.

    Returns
    -------
    Figure
        The figure object to persist or display
    """
    dflist = []
    # Clip target values above the 99th percentile of the data
    extremes = np.quantile(y, q=[0.01, 0.99])
    target = y[(y > extremes[0]) & (y < extremes[1])]
    cand_ = candidates or DEFAULT_CANDIDATES
    for label in cand_:
        params = getattr(scipy.stats, DIST_MAPPING[label]).fit(target)
        rv = getattr(scipy.stats, DIST_MAPPING[label])(*params)
        x = np.linspace(rv.ppf(0.01), rv.ppf(0.99))
        dflist.append(pd.DataFrame({"x": x, "y": rv.pdf(x), "dist": label}))

    tracedf = pd.concat(dflist, ignore_index=True)

    plt.ioff()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(target, stat="density", ax=ax)
    sns.lineplot(x="x", y="y", hue="dist", data=tracedf, palette="flare", ax=ax)

    fig.tight_layout()

    return fig

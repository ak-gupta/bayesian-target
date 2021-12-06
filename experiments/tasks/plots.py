"""Target distribution visualization."""

from typing import Dict

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
        fig = visualize_target_dist(data[metadata["target"]])
        fig.suptitle(f"{metadata['dataset_name']} target density")

    fig.savefig(OUTPUT_DIR / f"{metadata['dataset_name']}.png")

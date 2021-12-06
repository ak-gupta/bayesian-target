"""Flows for generating target distribution visualizations."""

from prefect import Flow, Parameter

from ..tasks import (
    read_data,
    read_metadata,
    render_dist_plot
)


def gen_visualization_flow() -> Flow:
    """Generate a flow to render distribution visualizations.
    
    Returns
    -------
    Flow
        Generated ``prefect.Flow``.
    """
    with Flow(name="Generate target distribution graph") as flow:
        dataset = Parameter("dataset", None)
        meta = read_metadata(dataset=dataset)
        data = read_data(metadata=meta)
        _ = render_dist_plot(data=data, metadata=meta)
    
    return flow

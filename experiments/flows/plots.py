"""Flows for generating target distribution visualizations."""

from prefect import Flow, Parameter
from rubicon_ml.workflow.prefect import get_or_create_project_task

from .. import OUTPUT_DIR
from ..tasks import (
    create_plot_df,
    parse_base_performance,
    parse_sampling_performance,
    read_data,
    read_metadata,
    render_dist_plot,
    render_sample_perf_plot
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


def gen_sample_performance_flow() -> Flow:
    """Generate a flow to render performance plots for sampling.
    
    Returns
    -------
    Flow
        Generated ``prefect.Flow``.
    """
    with Flow(name="Sampling performance plots") as flow:
        baseproj = get_or_create_project_task(
            "filesystem", str(OUTPUT_DIR), "Base Performance"
        )
        sampleproj = get_or_create_project_task(
            "filesystem", str(OUTPUT_DIR), "Number of samples"
        )
        basedf = parse_base_performance(project=baseproj)
        sampledf = parse_sampling_performance(project=sampleproj)
        plotdf = create_plot_df(base=basedf, exp=sampledf)

        render_sample_perf_plot(data=plotdf)
    
    return flow

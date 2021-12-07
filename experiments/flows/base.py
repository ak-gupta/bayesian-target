"""Flows for evaluating base model performance."""

from prefect import Flow, Parameter

from rubicon_ml.workflow.prefect import (
    get_or_create_project_task,
    create_experiment_task,
    log_metric_task,
    log_parameter_task
)

from .. import OUTPUT_DIR
from ..tasks import (
    read_data,
    read_metadata,
    fit_and_score_model,
    split_data
)


def gen_base_performance_flow() -> Flow:
    """Generate a flow to evaluate base model performance.
    
    Returns
    -------
    Flow
        Generated ``prefect.Flow``.
    """
    with Flow(name="Evaluate base model performance") as flow:
        project = get_or_create_project_task(
            "filesystem",
            OUTPUT_DIR,
            "Bayesian Target Encoding"
        )
        experiment = create_experiment_task(
            project, name="Base model performance"
        )

        dataset = Parameter("dataset", None)
        meta = read_metadata(dataset=dataset)
        data = read_data(metadata=meta)
        # Initialize the model
        # Score the model
        log_parameter_task(experiment, "dataset", dataset)
        # Log the scores

    return flow


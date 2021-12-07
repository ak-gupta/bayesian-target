"""Flows for evaluating base model performance."""

from prefect import Flow, Parameter, unmapped

from rubicon_ml.workflow.prefect import (
    get_or_create_project_task,
    create_experiment_task,
    log_metric_task,
    log_parameter_task,
)

from .. import OUTPUT_DIR
from ..tasks import (
    read_data,
    read_metadata,
    init_model,
    avg_fit_time,
    avg_score,
    fit_and_score_model,
    split_data,
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
            "filesystem", str(OUTPUT_DIR), "Bayesian Target Encoding"
        )
        experiment = create_experiment_task(project, name="Base model performance")

        dataset = Parameter("dataset", None)
        algorithm = Parameter("algorithm", None)
        meta = read_metadata(dataset=dataset)
        data = read_data(metadata=meta)
        model = init_model(algorithm=algorithm, metadata=meta)
        # Score the model
        splits = split_data(data=data, metadata=meta, estimator=model)
        scores = fit_and_score_model.map(
            data=unmapped(data),
            metadata=unmapped(meta),
            estimator=unmapped(model),
            splits=splits,
        )
        log_parameter_task(experiment, "dataset", dataset)
        log_parameter_task(experiment, "algorithm", algorithm)
        # Log scores
        final_score = avg_score(scoring_out=scores)
        final_fit_time = avg_fit_time(scoring_out=scores)
        log_metric_task(experiment, "score", final_score)
        log_metric_task(experiment, "fit-time", final_fit_time)

    return flow

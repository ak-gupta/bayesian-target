"""Flows for evaluating base model performance."""

from prefect import Flow, unmapped

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
    final_fit_times,
    final_scores,
    fit_and_score_model,
    split_data,
)


def gen_base_performance_flow(dataset: str, algorithm: str) -> Flow:
    """Generate a flow to evaluate base model performance.

    Parameters
    ----------
    dataset : str
        The name of the dataset.
    algorithm : {"linear", "xgboost", "lightgbm"}
        The modelling algorithm.

    Returns
    -------
    Flow
        Generated ``prefect.Flow``.
    """
    with Flow(name="Evaluate base model performance") as flow:
        project = get_or_create_project_task(
            "filesystem", str(OUTPUT_DIR), "Base Performance"
        )
        experiment = create_experiment_task(project, name=f"{dataset}-{algorithm}")
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
        final_score = final_scores(scoring_out=scores)
        final_fit_time = final_fit_times(scoring_out=scores)
        log_metric_task.map(
            unmapped(experiment), [f"score-{i}" for i in range(1, 11)], final_score
        )
        log_metric_task.map(
            unmapped(experiment), [f"fit-time-{i}" for i in range(1, 11)], final_fit_time
        )

    return flow

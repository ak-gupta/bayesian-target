"""Flow for evaluating the effect of samples on performance."""

from prefect import Flow, case, task, unmapped
from prefect.tasks.control_flow import merge

from rubicon_ml.workflow.prefect import (
    get_or_create_project_task,
    create_experiment_task,
    log_metric_task,
    log_parameter_task,
)

from .. import OUTPUT_DIR
from ..tasks import (
    positive_target,
    read_data,
    read_metadata,
    init_encoder,
    init_model,
    final_fit_times,
    final_scores,
    fit_and_score_model,
    fit_and_score_ensemble_model,
    split_data,
)

@task
def check_n_estimators(n_estimators: int) -> bool:
    """Check if the number of estimators is 0."""
    return n_estimators == 0

def gen_sampling_performance_flow(
    dataset: str, algorithm: str, n_estimators: int
) -> Flow:
    """Generate a flow to evaluate the effect of the number of samples on performance.

    Parameters
    ----------
    dataset : str
        The name of the dataset.
    algorithm : {"linear", "xgboost", "lightgbm"}
        The modelling algorithm.
    n_estimators : int
        The number of samples to use in the ensemble model.

    Returns
    -------
    Flow
        Generated ``prefect.Flow``
    """
    with Flow(name="Evaluate effect of number of samples") as flow:
        project = get_or_create_project_task(
            "filesystem", str(OUTPUT_DIR), "Number of samples"
        )
        experiment = create_experiment_task(
            project, name=f"{dataset}-{algorithm}-{n_estimators}"
        )
        meta = read_metadata(dataset=dataset)
        data = read_data(metadata=meta)
        finaldata = positive_target(data=data, metadata=meta)
        model = init_model(algorithm=algorithm, metadata=meta)
        encoder = init_encoder(algorithm="bayes", metadata=meta)
        # Score the model
        splits = split_data(data=finaldata, metadata=meta, estimator=model)
        cond = check_n_estimators(n_estimators)
        with case(cond, True):
            single_scores = fit_and_score_model.map(
                data=unmapped(data),
                metadata=unmapped(meta),
                estimator=unmapped(model),
                splits=splits,
                encoder=unmapped(encoder)
            )
        with case(cond, False):
            ensemble_scores = fit_and_score_ensemble_model.map(
                data=unmapped(finaldata),
                metadata=unmapped(meta),
                estimator=unmapped(model),
                splits=splits,
                encoder=unmapped(encoder),
                n_estimators=unmapped(n_estimators),
            )
        scores = merge(single_scores, ensemble_scores)
        log_parameter_task(experiment, "dataset", dataset)
        log_parameter_task(experiment, "algorithm", algorithm)
        log_parameter_task(experiment, "n_estimators", n_estimators)
        # Log scores
        final_score = final_scores(scoring_out=scores)
        final_fit_time = final_fit_times(scoring_out=scores)
        log_metric_task.map(
            unmapped(experiment), [f"score-{i}" for i in range(1, 6)], final_score
        )
        log_metric_task.map(
            unmapped(experiment), [f"fit-time-{i}" for i in range(1, 6)], final_fit_time
        )

    return flow

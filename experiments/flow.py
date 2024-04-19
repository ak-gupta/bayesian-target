"""Basic model fit and score flow."""

from lazyscribe.prefect import LazyProject
from prefect import Flow, case
from prefect.tasks.control_flow import merge

from experiments import OUTPUT_DIR, tasks


def gen_flow(
    project: str = "compare.json",
    dataset: str = "ames-housing",
    encoder: str = "bayes",
    algorithm: str = "xgboost",
    marginal: bool = False,
    residual: bool = False,
    seed: int = 42,
    n_estimators: int = 0,
) -> Flow:
    """Create a model fit and score flow.

    Parameters
    ----------
    project : str, optional (default "compare.json")
        The name of the project JSON file.
    dataset : str, optional (default "ames-housing")
        The name of the dataset.
    encoder : str, optional (default "bayes")
        The name of the categorical encoder.
    algorithm : str, optional (default "xgboost")
        The modelling algorithm.
    marginal : bool, optional (default False)
        Whether or not to use a marginal model output as the target for supervised
        encoding.
    residual : bool, optional (default False)
        Whether or not to use a residuals from the marginal model as the target for
        unsupervised encoding.

    Returns
    -------
    Flow
        The generated flow.
    """
    experiment_name = f"{dataset}-{encoder}-{algorithm}-{n_estimators}-{seed}"
    if marginal:
        experiment_name += "-marginal"
    elif residual:
        experiment_name += "-residual"
    init_project = LazyProject(fpath=OUTPUT_DIR / project, mode="w+", author="root")
    with Flow(name="Model fit") as flow:
        project = init_project()

        with project.log(name=experiment_name) as exp:
            # Log metadata
            exp.log_parameter("dataset", dataset)
            exp.log_parameter("encoder", encoder)
            exp.log_parameter("model", algorithm)
            exp.log_parameter("marginal", marginal)
            exp.log_parameter("residual", residual)
            exp.log_parameter("seed", seed)
            exp.log_parameter("n_estimators", n_estimators)
            # Retrieve metadata and build dataset
            meta = tasks.read_metadata(dataset=dataset)
            data = tasks.read_data(metadata=meta)

            estimator = tasks.init_model(algorithm=algorithm, metadata=meta, seed=seed)

            # Split and encode
            supervised = tasks.check_supervised(algorithm=encoder)
            encoder_object = tasks.init_encoder(
                algorithm=encoder, metadata=meta, residual=residual
            )
            train, test = tasks.split(data=data, metadata=meta, seed=seed)
            with case(supervised, True):
                fitted_encoder_super = tasks.fit_encoder(
                    data=train,
                    metadata=meta,
                    encoder=encoder_object,
                    estimator=estimator,
                    marginal=marginal,
                    residual=residual,
                )
            with case(supervised, False):
                fitted_encoder_unsup = tasks.fit_encoder(
                    data=data,
                    metadata=meta,
                    encoder=encoder_object,
                )
            fitted_encoder = merge(fitted_encoder_super, fitted_encoder_unsup)

            ensemble = tasks.check_ensemble(n_estimators=n_estimators)
            with case(ensemble, False):
                train_transformed = tasks.transform(
                    data=train, encoder=fitted_encoder, metadata=meta
                )
                test_transformed = tasks.transform(
                    data=test, encoder=fitted_encoder, metadata=meta
                )
                finaltrain = tasks.drop_nulls(data=train_transformed)
                finaltest = tasks.drop_nulls(data=test_transformed)
                # Fit and score
                fitted_estimator_std = tasks.train(
                    data=finaltrain, metadata=meta, estimator=estimator, seed=seed
                )
                score_std = tasks.score_model(
                    data=finaltest, metadata=meta, estimator=fitted_estimator_std
                )
            with case(ensemble, True):
                fitted_estimator_ens = tasks.train(
                    data=train,
                    metadata=meta,
                    estimator=estimator,
                    encoder=fitted_encoder,
                    n_estimators=n_estimators,
                    seed=seed,
                )
                score_ens = tasks.score_model(
                    data=test, metadata=meta, estimator=fitted_estimator_ens
                )

            score = merge(score_std, score_ens)
            exp.log_metric("score", score)

        project.save()

    return flow


def gen_sample_viz_flow(project: str = "sample.json") -> Flow:
    """Generate a flow to render sample performance visualizations.

    Parameters
    ----------
    project : str, optional (default "sample.json")
        The name of the project JSON file.

    Returns
    -------
    Flow
        The generated flow.
    """
    init_project = LazyProject(fpath=OUTPUT_DIR / project, mode="r", author="root")
    with Flow(name="Render sample visualizations") as flow:
        project = init_project()
        experiments, _ = project.to_tabular()
        df = tasks.project_to_df(data=experiments)
        _ = tasks.render_sample_perf_plot(data=df)

    return flow


def gen_comparison_viz_flow(project: str = "compare.json") -> Flow:
    """Generate a flow to render comparison performance visualizations.

    Parameters
    ----------
    project : str, optional (default "compare.json")
        The name of the project JSON file.

    Returns
    -------
    Flow
        The generated flow.
    """
    init_project = LazyProject(fpath=OUTPUT_DIR / project, mode="r", author="root")
    with Flow(name="Render compare visualizations") as flow:
        project = init_project()
        experiments, _ = project.to_tabular()
        df = tasks.project_to_df(data=experiments)
        _ = tasks.render_comparison_perf_plot(data=df)

    return flow

"""Run experiments.


I recommend suppressing logging from Prefect.

```console
$ export PREFECT__LOGGING__LEVEL=ERROR
```
"""

import click

from experiments.flow import gen_flow

@click.group()
def cli():
    """CLI group."""
    pass


@cli.command()
@click.option(
    "--dataset",
    type=click.Choice(
        [
            "ames-housing",
            "avocado-sales",
            "employee_salaries",
            "flight-delay-usa-dec-2017",
            "particulate-matter-ukair-2017",
            "churn",
            "click_prediction_small"
        ]
    ),
    help="The dataset"
)
@click.option(
    "--algorithm",
    type=click.Choice(["xgboost", "lightgbm", "gbm"]),
    help="The algorithm"
)
@click.option(
    "--n-estimators",
    type=click.INT,
    multiple=True,
    default=[0, 25, 50, 75, 100, 125, 150, 175, 200]
)
@click.option(
    "--seeds",
    type=click.INT,
    multiple=True,
    default=[5, 10, 16, 42, 44]
)
def sample(dataset, algorithm, n_estimators, seeds):
    """Run the sampling experiment."""
    for n_est in n_estimators:
        for seed in seeds:
            click.echo(
                click.style(
                    (
                        f"Running experiment for {dataset} with algorithm {algorithm}, "
                        f"{n_est} estimators, and seed {seed}"
                    ),
                    fg="green"
                )
            )
            flow = gen_flow(
                project="sample.json",
                dataset=dataset,
                encoder="bayes",
                algorithm=algorithm,
                seed=seed,
                n_estimators=n_est
            )
            _ = flow.run()
            if not _.is_successful():
                click.echo(click.style("Experiment failed.", fg="red"))
            else:
                click.echo(click.style("Experiment finished.", fg="green"))


@cli.command()
@click.option(
    "--dataset",
    type=click.Choice(
        [
            "ames-housing",
            "avocado-sales",
            "employee_salaries",
            "flight-delay-usa-dec-2017",
            "particulate-matter-ukair-2017",
            "churn",
            "click_prediction_small"
        ]
    ),
    help="The dataset"
)
@click.option(
    "--algorithm",
    type=click.Choice(["xgboost", "lightgbm", "gbm"]),
    help="The algorithm"
)
@click.option(
    "--encoder",
    type=click.Choice(["frequency", "glmm", "james-stein", "integer", "target", "bayes"]),
    help="Categorical encoder"
)
@click.option(
    "--n-estimators",
    type=click.INT,
    default=0
)
@click.option(
    "--seeds",
    type=click.INT,
    multiple=True,
    default=[5, 10, 16, 42, 44]
)
@click.option(
    "--marginal",
    is_flag=True,
    help="Whether or not to use marginal encoding"
)
@click.option(
    "--residual",
    is_flag=True,
    help="Whether or not to use residual encoding"
)
def compare(dataset, algorithm, encoder, n_estimators, seeds, marginal, residual):
    """Run the comparison experiment."""
    for seed in seeds:
        click.echo(
            click.style(
                (
                    f"Running experiment for {dataset} with algorithm {algorithm}, "
                    f"encoder {encoder}, {n_estimators} estimators, and seed {seed}."
                ),
                fg="green"
            )
        )
        flow = gen_flow(
            project="compare.json",
            dataset=dataset,
            encoder=encoder,
            algorithm=algorithm,
            marginal=marginal,
            residual=residual,
            seed=seed,
            n_estimators=n_estimators,
        )
        _ = flow.run()
        if not _.is_successful():
            click.echo(click.style("Experiment failed.", fg="red"))
        else:
            click.echo(click.style("Experiment finished.", fg="green"))


if __name__ == "__main__":
    cli()

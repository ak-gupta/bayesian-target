"""Run experiments.


I recommend suppressing logging from Prefect.

```console
$ export PREFECT__LOGGING__LEVEL=ERROR
```
"""

import click

from experiments.flows import gen_base_performance_flow, gen_sampling_performance_flow


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
            "nyc-taxi-green-dec-2016",
            "particulate-matter-ukair-2017"
        ]
    ),
    help="The dataset"
)
def base(dataset):
    """Run the base performance experiment."""
    for algo in ["xgboost", "lightgbm", "gbm"]:
        for seed in [5, 10, 16, 42, 44]:
            click.echo(
                click.style(
                    f"Running experiment for {dataset} with algorithm {algo} and seed {seed}",
                    fg="green"
                )
            )
            flow = gen_base_performance_flow(dataset=dataset, algorithm=algo, seed=seed)
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
            "nyc-taxi-green-dec-2016",
            "particulate-matter-ukair-2017"
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
def sample(dataset, algorithm, n_estimators):
    """Run the sampling experiment."""
    for n_est in n_estimators:
        for seed in [5, 10, 16, 42, 44]:
            click.echo(
                click.style(
                    (
                        f"Running experiment for {dataset} with algorithm {algorithm}, "
                        f"{n_est} estimators, and seed {seed}"
                    ),
                    fg="green"
                )
            )
            flow = gen_sampling_performance_flow(
                dataset=dataset, algorithm=algorithm, n_estimators=n_est, seed=seed
            )
            _ = flow.run()
            if not _.is_successful():
                click.echo(click.style("Experiment failed.", fg="red"))
            else:
                click.echo(click.style("Experiment finished.", fg="green"))

if __name__ == "__main__":
    cli()

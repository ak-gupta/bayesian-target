"""Run experiments."""

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
@click.option(
    "--algorithm",
    type=click.Choice(["linear", "xgboost", "lightgbm"]),
    help="The modelling algorithm"
)
def base(dataset, algorithm):
    """Run the base performance experiment."""
    flow = gen_base_performance_flow(dataset=dataset, algorithm=algorithm)
    flow.run()


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
    type=click.Choice(["linear", "xgboost", "lightgbm"]),
    help="The modelling algorithm"
)
@click.option(
    "--n-estimators",
    type=click.INT,
    help="The number of estimators to use"
)
def sample(dataset, algorithm, n_estimators):
    """Run the sampling experiment."""
    flow = gen_sampling_performance_flow(
        dataset=dataset, algorithm=algorithm, n_estimators=n_estimators
    )
    flow.run()

if __name__ == "__main__":
    cli()

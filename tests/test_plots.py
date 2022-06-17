"""Test the target visualization utility."""

from bayte.plots import visualize_target_dist

def test_plot(toy_regression_dataset):
    """Test visualizing the target distribution."""
    _, y = toy_regression_dataset
    fig = visualize_target_dist(y)

    ax = fig.gca()

    assert len(ax.lines) == 8

    labels = [line.get_label() for line in ax.lines]
    assert labels[4:] == ["exponential", "gamma", "invgamma", "normal"]

"""Import path."""

__version__ = "0.1.0"
__description__ = "Bayesian target encoding with scikit-learn and scipy"

from typing import List

from bayesian_target.encoder import BayesianTargetEncoder
from bayesian_target.ensemble import BayesianTargetEstimator

__all__: List[str] = ["BayesianTargetEncoder", "BayesianTargetEstimator"]

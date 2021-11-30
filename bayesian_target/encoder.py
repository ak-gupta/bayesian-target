"""Bayesian target encoder."""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

from joblib import Parallel, effective_n_jobs
import numpy as np
import scipy.stats
from sklearn.preprocessing._encoders import _BaseEncoder
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import check_is_fitted

LOG = logging.getLogger(__name__)


def _init_prior(dist: str, y) -> Tuple:
    """Initialize the prior distribution based on the input likelihood.
    
    Parameters
    ----------
    dist : {"bernoulli", "exponential", "gamma", "invgamma"}
        The likelihood for the target.
    y : array-like of shape (n_samples,)
        Target values.
    
    Returns
    -------
    tuple
        The initialization parameters.
    """
    if dist == "bernoulli":
        return np.average(y), 1 - np.average(y)
    elif dist == "exponential":
        return y.shape[0] + 1, np.sum(y)
    elif dist in ("gamma", "invgamma"):
        fitter = getattr(scipy.stats, dist)
        alpha, _, _ = fitter.fit(y)
        
        return y.shape[0] * alpha, 0, np.sum(y)
    else:
        raise NotImplementedError(f"Likelihood {dist} has not been implemented.")


def _update_posterior(y, mask, dist, params) -> Tuple:
    """Generate the parameters for the posterior distribution.
    
    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target values.
    mask : array-like of shape (n_samples,)
        A boolean array indicating the observations in ``y`` that should
        be used to generate the posterior distribution.
    dist : {"bernoulli", "exponential", "gamma", "invgamma"}
        The likelihood for the target.
    params : Tuple
        The prior distribution parameters.

    Returns
    -------
    tuple
        Parameters for the posterior distribution. The parameters are based on the
        ``scipy.stats`` parameterization of the posterior.
    
    References
    ----------
    .. [1] A compendium of conjugate priors, from https://www.johndcook.com/CompendiumOfConjugatePriors.pdf
    """
    if dist == "bernoulli":
        return params[0] + np.sum(y[mask]), params[1] + np.sum(mask) - np.sum(y[mask]), 0, 1
    elif dist == "exponential":
        return params[0] + np.sum(mask), 0, params[1]/(1 + params[1] * np.sum(y[mask]))
    elif dist in ("gamma", "invgamma"):
        fitter = getattr(scipy.stats, dist)
        alpha, _, _ = fitter.fit(y)

        return np.sum(mask) * alpha + params[0], 0, params[2]/(1 + np.sum(y[mask]))
    else:
        raise NotImplementedError(f"Likelihood {dist} has not been implemented.")

_POSTERIOR_DISPATCHER: Dict[str, Callable] = {
    "bernoulli": scipy.stats.beta,
    "exponential": scipy.stats.gamma,
    "gamma": scipy.stats.gamma,
    "invgamma": scipy.stats.invgamma
}

class BayesianTargetEncoder(_BaseEncoder):
    """Bayesian target encoder.

    This encoder will

    1. Derive the prior distribution from the supplied ``dist``,
    2. Initialize the prior distribution hyperparameters using the training data,
    3. For each level in each categorical,
        * Generate the posterior distribution,
        * Set the encoding value(s) as a sample or the mean from the posterior distribution
    
    Parameters
    ----------
    dist : {"bernoulli", "exponential", "gamma", "invgamma"}
        The likelihood for the target.
    sample : bool, optional (default False)
        Whether or not to encode the categorical values as a sample from the posterior
        distribution or the mean.
    categories : 'auto' or list of array-like, optional (default 'auto')
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values within a single feature, and should be sorted in case of
          numeric values.

        The used categories can be found in the ``categories_`` attribute.
    initializer : callable, optional (default None)
        A callback function for returning the prior distribution hyperparameters.
        This function must take in the ``dist`` value and the target array.
    dtype : number type, optional (default float)
        Desired dtype of output.
    handle_unknown : {'error', 'ignore'}, optional (default "ignore")
        Whether to raise an error or ignore if an unknown categorical feature
        is present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered, the resulting
        encoding will be taken from the prior distribution.
    n_jobs : int, optional (default None)
        The number of cores to run in parallel when fitting the encoder.
        ``None`` means 1 unless in a ``joblib.parallel_backend`` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    prior_params_ : tuple
        The estimated hyperparameters for the prior distribution.
    posterior_params_ : list
        A list of lists. Each entry in the list corresponds to the categorical
        feature in ``categories_``. Each index in the nested list contains
        the parameters for the posterior distribution for the given level.
    """

    _required_parameters = ["dist"]

    def __init__(
        self,
        dist: str,
        sample: bool = False,
        categories: Union[str, List] = "auto",
        initializer: Optional[Callable] = None,
        dtype=np.float64,
        handle_unknown: str = "ignore",
        n_jobs: Optional[int] = None,
    ):
        """Init method."""
        self.dist = dist
        self.sample = sample
        self.categories = categories
        self.initializer = initializer
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.n_jobs = n_jobs

    
    def fit(self, X, y):
        """Fit the bayesian target encoder.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature and the posterior
            distributions.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
        
        Returns
        -------
        self : object
            Fitted encoder.
        """
        X, y = self._validate_data(X, y, dtype=None)
        self._fit(X, handle_unknown=self.handle_unknown, force_all_finite=True)
        # Initialize the prior distribution parameters
        initializer_ = self.initializer or _init_prior
        self.prior_params_ = initializer_(self.dist, y)

        if effective_n_jobs(self.n_jobs) == 1:
            parallel, fn = list, _update_posterior
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
            fn = delayed(_update_posterior)
        
        LOG.info("Determining the posterior distribution parameters...")
        self.posterior_params_ = []
        for index, cat in enumerate(self.categories_):
            self.posterior_params_.append(
                parallel(
                    fn(y, X[:, index] == level, self.dist, self.prior_params_)
                    for level in cat
                )
            )

        return self


    def transform(self, X):
        """Transform the input dataset.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.
        
        Returns
        -------
        ndarray
            Transformed input.
        """
        check_is_fitted(self)

        X_int, X_mask = self._transform(
            X,
            handle_unknown=self.handle_unknown,
            force_all_finite=True,
        )

        X_out = np.zeros(X.shape)
        # Loop through each categorical
        for idx, cat in enumerate(self.categories_):
            # Loop through each level and sample or evaluate the mean from the posterior
            for levelno in range(cat.shape[0]):
                mask = (X_int[:, idx] == levelno) & (X_mask[:, idx])
                rv = _POSTERIOR_DISPATCHER[self.dist](*self.posterior_params_[idx][levelno])
                if self.sample:
                    X_out[mask, idx] = rv.rvs(size=np.sum(mask))
                else:
                    X_out[mask, idx] = rv.moment(n=1)
            # Capture any new levels
            mask = (~X_mask[:, idx])
            rv = _POSTERIOR_DISPATCHER[self.dist](*self.prior_params_)
            if self.sample:
                X_out[mask, idx] = rv.rvs(size=np.sum(mask))
            else:
                X_out[mask, idx] = rv.moment(n=1)

        return X_out

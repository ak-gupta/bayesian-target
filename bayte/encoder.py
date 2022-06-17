"""Bayesian target encoder."""

import logging
from typing import Callable, List, Optional, Tuple, Union

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
    dist : {"bernoulli", "exponential", "gamma", "invgamma", "normal"}
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
    elif dist == "multinomial":
        _, counts = np.unique(y, return_counts=True)

        return tuple(counts)
    elif dist == "exponential":
        return y.shape[0] + 1, np.sum(y)
    elif dist == "normal":
        # First parameter is the sample mean
        mean = np.average(y)

        return mean, 1 / np.sum(1 / np.square(y - mean))
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
    .. [1] A compendium of conjugate priors, from https://www.johndcook.com/CompendiumOfConjugatePriors.pdf  # noqa: E501
    """
    if dist == "bernoulli":
        return (
            params[0] + np.sum(y[mask]),
            params[1] + np.sum(mask) - np.sum(y[mask]),
            0,
            1,
        )
    elif dist == "multinomial":
        # This assumes the classes are 0, 1, 2, ..., m
        unique, counts = np.unique(y[mask], return_counts=True)
        class_counts = np.zeros((len(params),))
        class_counts[unique] = counts

        return tuple(np.array(params) + class_counts)
    elif dist == "exponential":
        return (
            params[0] + np.sum(mask),
            0,
            params[1] / (1 + params[1] * np.sum(y[mask])),
        )
    elif dist == "normal":
        # Known variance is the non-sample variance from the training data
        var = np.var(y)

        factor = 1 / ((1 / params[1]) + (np.sum(mask) / var))

        return factor * ((params[0] / params[1]) + (np.sum(y[mask]) / var)), np.sqrt(
            factor
        )
    elif dist in ("gamma", "invgamma"):
        fitter = getattr(scipy.stats, dist)
        alpha, _, _ = fitter.fit(y)

        return np.sum(mask) * alpha + params[0], 0, params[2] / (1 + np.sum(y[mask]))
    else:
        raise NotImplementedError(f"Likelihood {dist} has not been implemented.")


def _encode_level(mask, dist, sample, params, random_state):
    """Encode a given level.

    Parameters
    ----------
    mask : array of shape (n_samples,)
        A boolean mask for the categorical value.
    level : int
        The level to encode.
    dist : str
        The likelihood.
    sample : bool
        Whether to sample or take the first moment from the posterior distribution.
    params : tuple
        Posterior parameters.
    random_state : int or None
        An optional random state for reproducible results

    Returns
    -------
    np.ndarray
        The array of encoded values. The array is n_samples long; the width
        depends on the number of values produced by the posterior distribution.
        Unencoded values are labeled as 0.
    """
    if dist == "bernoulli":
        random_var = scipy.stats.beta(*params)
    elif dist == "multinomial":
        random_var = scipy.stats.dirichlet(params)
    elif dist == "normal":
        random_var = scipy.stats.norm(*params)
    elif dist in ("exponential", "gamma", "invgamma"):
        random_var = scipy.stats.gamma(*params)
    else:
        raise NotImplementedError(f"Likelihood {dist} has not been implemented")

    if sample:
        if random_state is not None:
            random_var.random_state = np.random.Generator(np.random.PCG64(random_state))
        encoding = random_var.rvs(size=1).ravel()
    else:
        avg = random_var.mean()
        if isinstance(avg, float):
            encoding = np.array([avg])  # Univariate distributions provide a float
        else:
            encoding = avg  # Multinomial provides an array

    mask_options = [0, 1, 999]
    for opt in mask_options:
        if (encoding == opt).sum() == 0:
            mask_val = opt
            break
    else:
        raise ValueError("Unable to set a mask value.")

    X_out = np.empty((mask.shape[0], len(encoding)), dtype=np.float64)
    X_out.fill(mask_val)
    X_out[mask, :] = encoding
    X_mask = np.ma.masked_equal(X_out, mask_val)

    return X_mask


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
    dist : {"bernoulli", "multinomial", "exponential", "gamma", "invgamma", "normal"}
        The likelihood for the target.

        .. important::

            For the gamma distribution, we assume a *known* shape parameter alpha. For the
            normal distribution, we assume a *known* variance. Both are estimated directly
            from the training data.

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
    chunksize : int, optional (default None)
        The number of categorical levels to combine at one time when calling
        ``transform``. Increasing the chunksize will increase memory usage. By
        default, all categoricals will be combined in a single step.
    random_state : int, optional (default None)
        An optional random state for scipy sampling.

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
        chunksize: int = 10,
        random_state: Optional[int] = None,
    ):
        """Init method."""
        self.dist = dist
        self.sample = sample
        self.categories = categories
        self.initializer = initializer
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.random_state = random_state

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

        if effective_n_jobs(self.n_jobs) == 1:
            parallel, fn = list, _encode_level
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
            fn = delayed(_encode_level)

        encoded = []
        for idx, cat in enumerate(self.categories_):
            LOG.debug(
                f"Running transform for categorical {idx} with {cat.shape[0]} levels"
            )
            # Get the masked array for each level
            varencoded = parallel(
                fn(
                    (X_int[:, idx] == levelno) & (X_mask[:, idx]),
                    self.dist,
                    self.sample,
                    self.posterior_params_[idx][levelno],
                    self.random_state,
                )
                for levelno in range(cat.shape[0])
                if np.sum(X_int[:, idx] == levelno) > 0
            )
            # Add new categorical encodings
            if np.sum(~X_mask[:, idx]) > 0:
                LOG.warning(
                    f"Found {np.sum(~X_mask[:, idx])} rows with novel levels for "
                    f"categorical variable {idx}"
                )

                varencoded.append(
                    _encode_level(
                        (~X_mask[:, idx]),
                        self.dist,
                        self.sample,
                        self.prior_params_,
                        self.random_state,
                    )
                )

            # Combine each chunk before combining everything at the end
            while len(varencoded) > 2:
                if self.chunksize is None:
                    n_chunks = 1
                else:
                    n_chunks = np.ceil(len(varencoded) / self.chunksize)
                chunks = np.array_split(np.arange(len(varencoded)), n_chunks)

                varencoded = list(
                    np.ma.stack(varencoded[chunk[0] : chunk[-1] + 1], axis=2).sum(
                        axis=2
                    )
                    for chunk in chunks
                )

            combined = np.ma.stack(varencoded, axis=2).sum(axis=2)
            encoded.append(combined.data)

        return np.hstack(encoded)

"""BayesianTargetEstimator.

Ensemble estimator that creates multiple models through sampling.
"""

from copy import deepcopy
import logging
from typing import List, Optional, Tuple, Union

from joblib import Parallel, effective_n_jobs
import numpy as np
from pandas.api.types import is_categorical_dtype
from sklearn.base import (
    ClassifierMixin,
    RegressorMixin,
    clone,
    is_classifier,
)
from sklearn.ensemble._base import BaseEnsemble
from sklearn.utils import check_random_state
from sklearn.utils.fixes import delayed
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted

LOG = logging.getLogger(__name__)


def _sample_and_fit(
    estimator, encoder, X, y, categorical_feature, random_state, **fit_params
):
    """Sample and fit the estimator.

    Parameters
    ----------
    estimator : estimator object
        The base estimator.
    encoder : estimator object
        The fitted Bayesian target encoder.
    X : array-like of shape (n_samples, n_features)
        The data to determine the categories of each feature and
        the posterior distributions.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    categorical_feature : list
        A boolean mask indicating which columns are categorical
    random_state : int or None
        An optional random state for the sampling
    **fit_params
        Parameters to be passed to the underlying estimator.

    Returns
    -------
    estimator
        The trained estimator.
    """
    if random_state is not None:
        encoder.set_params(random_state=random_state)
    X_encoded = encoder.transform(X[:, categorical_feature])
    X_sample = np.hstack((X[:, ~categorical_feature], X_encoded))

    return estimator.fit(X_sample, y, **fit_params)


class BaseSamplingEstimator(BaseEnsemble):
    """Base bayesian target encoding sampling estimator.

    This estimator will use the bayesian target encoder to encode multiple
    training datasets. The supplied estimator will be trained multiple times,
    producing ``n_estimators`` submodels. The prediction from the model will
    be an average of each submodel's output.

    Parameters
    ----------
    base_estimator : object
        The base estimator from which the ensemble is built.
    encoder : BayesianTargetEncoder
        A bayesian target encoder object.
    n_estimators : int, optional (default 10)
        The number of estimators to train.
    n_jobs : int, optional (default None)
        The number of cores to run in parallel when fitting the encoder.
        ``None`` means 1 unless in a ``joblib.parallel_backend`` context.
        ``-1`` means using all processors.
    random_state : int, optional (default None)
        Random seed used for generating random seeds for sampling.
    estimator_params : list of str, optional (default tuple())
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    Attributes
    ----------
    categorical_ : np.ndarray
        A boolean mask indicating which columns are categorical and which are continuous.
    base_estimator_ : estimator object
        The base estimator from which the ensemble is grown.
    estimators_ : list
        The collection of fitted base estimators.
    """

    _required_parameters = ["base_estimator", "encoder"]

    def __init__(
        self,
        base_estimator,
        encoder,
        n_estimators: int = 10,
        n_jobs: Optional[int] = None,
        random_state: int = None,
        estimator_params: Union[List[str], Tuple] = tuple(),
    ):
        """Init method."""
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params
        self.encoder = encoder
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(
        self,
        X,
        y,
        categorical_feature: Union[List[str], List[int], str] = "auto",
        **fit_params
    ):
        """Fit the estimator.

        Fitting the estimator involves

        1. Fitting the encoder,
        2. Sampling the encoder ``n_estimators`` times,
        3. Fitting the submodels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature and the posterior
            distributions.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        categorical_feature : list or str, optional (default "auto")
            Categorical features to encode. If a list of int, it will be interpreted
            as indices. If a list of string, it will be interpreted as the column names
            in a pandas DataFrame. If "auto" and the data is a pandas DataFrame, any columns
            with a ``pd.Categorical`` data type will be encoded. A numpy array with "auto"
            will result in all input features being treated as categorical.
        **fit_params
            Parameters to be passed to the underlying estimator.

        Returns
        -------
        self
            The trained estimator.
        """
        rng = check_random_state(self.random_state)
        self.rstates_ = rng.randint(self.n_estimators * 10, size=self.n_estimators)
        # Get the categorical columns
        if hasattr(X, "columns"):
            self.categorical_ = np.zeros(X.shape[1], dtype=bool)
            for idx, col in enumerate(X.columns):
                if categorical_feature == "auto":
                    if is_categorical_dtype(X[col]):
                        self.categorical_[idx] = True
                elif col in categorical_feature:
                    self.categorical_[idx] = True

        if is_classifier(self.base_estimator):
            check_classification_targets(y)
            self.classes_ = np.unique(y)

        X, y = self._validate_data(X, y, dtype=None)

        if not hasattr(self, "categorical_"):
            if categorical_feature == "auto":
                LOG.warning(
                    "No categorical features provided. All features will be treated as categorical."
                )
                self.categorical_ = np.ones(X.shape[1], dtype=bool)
            else:
                self.categorical_ = np.zeros(X.shape[1], dtype=bool)
                for col in categorical_feature:
                    self.categorical_[col] = True

        # Fit the encoder
        if hasattr(self.encoder, "posterior_params_"):
            LOG.warning("Supplied with a fitted encoder. Not re-fitting.")
            self.encoder_ = deepcopy(self.encoder)
            self.encoder_.set_params(sample=True)
        else:
            LOG.info("Fitting the target encoder.")
            self.encoder_ = clone(self.encoder)
            self.encoder_.set_params(sample=True)
            self.encoder_.fit(
                X[:, self.categorical_], y
            )  # Need to filter the columns to categoricals

        self._validate_estimator()
        if effective_n_jobs(self.n_jobs) == 1:
            parallel, fn = list, _sample_and_fit
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
            fn = delayed(_sample_and_fit)

        LOG.info("Training the estimator(s).")
        self.estimators_ = parallel(
            fn(
                clone(self.base_estimator),
                deepcopy(self.encoder_),
                X,
                y,
                self.categorical_,
                self.rstates_[idx],
                **fit_params
            )
            for idx in range(self.n_estimators)
        )

        return self


class BayesianTargetRegressor(RegressorMixin, BaseSamplingEstimator):
    """Sampling bayesian target regressor.

    This estimator will use the bayesian target encoder to encode multiple
    training datasets. The supplied estimator will be trained multiple times,
    producing ``n_estimators`` submodels. The prediction from the model will
    be an average of each submodel's output.

    Parameters
    ----------
    base_estimator : object
        The base estimator from which the ensemble is built.
    encoder : BayesianTargetEncoder
        A bayesian target encoder object.
    n_estimators : int, optional (default 10)
        The number of estimators to train.
    n_jobs : int, optional (default None)
        The number of cores to run in parallel when fitting the encoder.
        ``None`` means 1 unless in a ``joblib.parallel_backend`` context.
        ``-1`` means using all processors.
    estimator_params : list of str, optional (default tuple())
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    Attributes
    ----------
    categorical_ : np.ndarray
        A boolean mask indicating which columns are categorical and which are continuous.
    base_estimator_ : estimator object
        The base estimator from which the ensemble is grown.
    estimators_ : list
        The collection of fitted base estimators.
    """

    @if_delegate_has_method(delegate="base_estimator")
    def predict(self, X):
        """Call predict on the estimators.

        The output of this function is the average prediction from all submodels.
        The function will encode the categorical variables using the mean of the posterior
        distribution.

        Parameters
        ----------
        X : indexable, length (n_samples,)
            Must fulfill the input assumptions of ``fit``.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The predicted values for ``X`` based on the average from each submodel.
        """
        check_is_fitted(self)
        self.encoder_.set_params(sample=False)

        X = check_array(X, dtype=None)
        X_encoded = self.encoder_.transform(X[:, self.categorical_])
        X_predict = np.hstack((X[:, ~self.categorical_], X_encoded))

        # Predict
        parallel = Parallel(n_jobs=self.n_jobs)

        out = parallel(delayed(model.predict)(X_predict) for model in self.estimators_)
        out = np.asarray(out)

        return np.average(out, axis=0)


class BayesianTargetClassifier(ClassifierMixin, BaseSamplingEstimator):
    """Sampling bayesian target classifier.

    This estimator will use the bayesian target encoder to encode multiple
    training datasets. The supplied estimator will be trained multiple times,
    producing ``n_estimators`` submodels. The predicted labels will either be
    based on majority rule voting (``voting="hard"``) or using the argmax of
    the sums of predicted probabilities (``voting="soft"``).

    Voting implementation from `scikit-learn <https://github.com/scikit-learn/scikit-learn/blob/e707e61d85127dae1f58e9ad2689ca2708add0a4/sklearn/ensemble/_voting.py#L148>`_  # noqa: E501

    Parameters
    ----------
    base_estimator : object
        The base estimator from which the ensemble is built.
    encoder : BayesianTargetEncoder
        A bayesian target encoder object.
    n_estimators : int, optional (default 10)
        The number of estimators to train.
    voting : {"hard", "soft"}, optional (default "hard")
        If "hard", uses predicted class labels for majority rule voting.
        If "soft", predicts class label based on argmax of the sums of
        predicted probabilities, which is recommended for an ensemble of
        well-calibrated classifiers.
    n_jobs : int, optional (default None)
        The number of cores to run in parallel when fitting the encoder.
        ``None`` means 1 unless in a ``joblib.parallel_backend`` context.
        ``-1`` means using all processors.
    random_state : int, optional (default None)
        Random seed used for generating random seeds for sampling.
    estimator_params : list of str, optional (default tuple())
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    Attributes
    ----------
    categorical_ : np.ndarray
        A boolean mask indicating which columns are categorical and which are continuous.
    base_estimator_ : estimator object
        The base estimator from which the ensemble is grown.
    estimators_ : list
        The collection of fitted base estimators.
    """

    _required_parameters = ["base_estimator", "encoder"]

    def __init__(
        self,
        base_estimator,
        encoder,
        n_estimators: int = 10,
        voting: str = "hard",
        n_jobs: Optional[int] = None,
        random_state: int = None,
        estimator_params: Union[List[str], Tuple] = tuple(),
    ):
        """Init method."""
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.voting = voting
        self.encoder = encoder
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.estimator_params = estimator_params

    @if_delegate_has_method(delegate="base_estimator")
    def predict(self, X):
        """Predict class labels for X.

        Parameters
        ----------
        X : indexable, length (n_samples,)
            Must fulfill the assumptions of ``fit``.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self)
        self.encoder_.set_params(sample=False)

        X = check_array(X, dtype=None)
        X_encoded = self.encoder_.transform(X[:, self.categorical_])
        X_predict = np.hstack((X[:, ~self.categorical_], X_encoded))

        # Predict
        parallel = Parallel(n_jobs=self.n_jobs)

        if self.voting == "soft":
            out = parallel(
                delayed(model.predict_proba)(X_predict) for model in self.estimators_
            )
            out = np.asarray(out)
            avg = np.average(out, axis=0)

            vote = np.argmax(avg, axis=1)
        elif self.voting == "hard":
            out = parallel(
                delayed(model.predict)(X_predict) for model in self.estimators_
            )
            out = np.asarray(out)
            vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x)), axis=0, arr=out
            )

        return vote

    @if_delegate_has_method(delegate="base_estimator")
    def predict_proba(self, X):
        """Call predict_proba on the estimators.

        The output of this function is the average prediction from all submodels.
        The function will encode the categorical variables using the mean of the posterior
        distribution.

        Parameters
        ----------
        X : indexable, length (n_samples,)
            Must fulfill the input assumptions of ``fit``.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The predicted class probabilities for ``X`` based on the average from each submodel.
        """
        check_is_fitted(self)
        self.encoder_.set_params(sample=False)

        X = check_array(X, dtype=None)
        X_encoded = self.encoder_.transform(X[:, self.categorical_])
        X_predict = np.hstack((X[:, ~self.categorical_], X_encoded))

        # Predict
        parallel = Parallel(n_jobs=self.n_jobs)

        out = parallel(
            delayed(model.predict_proba)(X_predict) for model in self.estimators_
        )
        out = np.asarray(out)

        return np.average(out, axis=0)

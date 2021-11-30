"""BayesianTargetEstimator.

Ensemble estimator that creates multiple models through sampling.
"""

from typing import List, Optional, Tuple, Union

from joblib import Parallel, effective_n_jobs
import numpy as np
from pandas.api.types import is_categorical_dtype
from sklearn.base import clone
from sklearn.ensemble._base import BaseEnsemble
from sklearn.utils.fixes import delayed
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_array, check_is_fitted


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator."""
    def check(self):
        if hasattr(self, "base_estimator"):
            getattr(self.base_estimator, attr)

            return True
        getattr(self.base_estimator, attr)

        return True
    
    return check



def _sample_and_fit(estimator, encoder, X, y, categorical_feature, **fit_params):
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
    **fit_params
        Parameters to be passed to the underlying estimator.
    
    Returns
    -------
    estimator
        The trained estimator.
    """
    X_sample = X.copy()
    X_encoded = encoder.transform(X_sample[:, categorical_feature])
    for idx, col in enumerate(categorical_feature):
        if not col:
            continue
        X_sample[:, idx] = X_encoded[:, idx]
    
    return estimator.fit(X_sample, y, **fit_params)

class BayesianTargetEstimator(BaseEnsemble):
    """Bayesian target estimator.
    
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
    base_estimator_
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
        estimator_params: Union[List[str], Tuple] = tuple()
    ):
        """Init method."""
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
        )
        self.encoder = encoder
        self.n_jobs = n_jobs


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
            with a ``pd.Categorical`` data type will be encoded. "auto" with a numpy array will
            be ignored.
        **fit_params
            Parameters to be passed to the underlying estimator.
        
        Returns
        -------
        self
            The trained estimator.
        """
        # Get the categorical columns
        if hasattr(X, "columns"):
            self.categorical_ = np.zeros(X.shape[1], dtype=bool)
            for idx, col in enumerate(X.columns):
                if categorical_feature == "auto":
                    if is_categorical_dtype(X[col]):
                        self.categorical_[idx] = True
                elif col in categorical_feature:
                    self.categorical_[idx] = True

        X, y = self._validate_data(X, y, dtype=None)

        if not hasattr(self, "categorical_") and categorical_feature != "auto":
            self.categorical_ = np.zeros(X.shape[1], dtype=bool)
            for col in categorical_feature:
                self.categorical_[col] = True

        # Fit the encoder
        self.encoder_ = clone(self.encoder)
        self.encoder_.set_params(sample=True)
        self.encoder_.fit(X[:, self.categorical_], y)  # Need to filter the columns to categoricals

        self._validate_estimator()
        self.estimators_ = []
        estimators = [self._make_estimator() for _ in range(self.n_estimators)]

        if effective_n_jobs(self.n_jobs) == 1:
            parallel, fn = list, _sample_and_fit
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
            fn = delayed(_sample_and_fit)
        
        parallel(
            fn(estimator, self.encoder_, X, y, self.categorical_, **fit_params)
            for estimator in estimators
        )

        return self


    @available_if(_estimator_has("predict"))
    def predict(self, X) -> np.ndarray:
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

        X_copy = check_array(X, dtype=None)
        X_encoded = self.encoder_.transform(X_copy[:, self.categorical_])
        for idx, col in enumerate(self.categorical_):
            if not col:
                continue
            X_copy[:, idx] = X_encoded[:, idx]
        
        # Predict
        parallel = Parallel(n_jobs=self.n_jobs)

        out = parallel(
            delayed(model.predict)(X_copy) for model in self.estimators_
        )

        return np.average(np.vstack(out), axis=0)


    @available_if(_estimator_has("predict"))
    def predict_proba(self, X) -> np.ndarray:
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

        X_copy = check_array(X, dtype=None)
        X_encoded = self.encoder_.transform(X_copy[:, self.categorical_])
        for idx, col in enumerate(self.categorical_):
            if not col:
                continue
            X_copy[:, idx] = X_encoded[:, idx]
        
        # Predict
        parallel = Parallel(n_jobs=self.n_jobs)

        out = parallel(
            delayed(model.predict_proba)(X_copy) for model in self.estimators_
        )

        return np.average(np.vstack(out), axis=0)


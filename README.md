
[![codecov](https://codecov.io/github/ak-gupta/bayte/branch/main/graph/badge.svg?token=S8BUVKF37O)](https://codecov.io/github/ak-gupta/bayte) [![Maintainability](https://api.codeclimate.com/v1/badges/5c0b77d0e9b8f899ee95/maintainability)](https://codeclimate.com/github/ak-gupta/bayte/maintainability) ![PyPI](https://img.shields.io/pypi/v/bayte) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bayte) [![Documentation Status](https://readthedocs.org/projects/bayte/badge/?version=latest)](https://bayte.readthedocs.io/en/latest/?badge=latest)

# Overview

This package is a lightweight implementation of bayesian target encoding. This implementation is taken
from [Slakey et al.](https://arxiv.org/pdf/1904.13001.pdf), with ensemble methodology from [Larionov](https://arxiv.org/pdf/2006.01317.pdf).

The encoding proceeds as follows:

1. User observes and chooses a likelihood for the target variable (e.g. Bernoulli for a binary classification problem),
2. Using [Fink's Compendium of Priors](https://www.johndcook.com/CompendiumOfConjugatePriors.pdf), derive the conjugate prior for the likelihood (e.g. Beta),
3. Use the training data to initialize the hyperparameters for the prior distribution
    * **NOTE**: This process is generally reliant on common interpretations of hyperparameters.
4. Using Fink's Compendium, derive the methodology for generating the posterior distribution,
5. For each level in the categorical variable,
    1. Generate the posterior distribution using the observed target values for the categorical level,
    2. Set the encoding value to a sample from the posterior distribution
        * If a new level has appeared in the dataset, the encoding will be sampled from the prior distribution.
          To disable this behaviour, initialize the encoder with ``handle_unknown="error"``.

Then, we repeat step 5.2 a total of ``n_estimators`` times, generating a total of ``n_estimators`` training datasets
with unique encodings. The end model is a vote from each sampled dataset.

For reproducibility, you can set the encoding value to the mean of the posterior distribution instead.

## Installation

```console
python -m pip install bayte@git+https://github.com/ak-gupta/bayte
```

## Usage

### Encoding

Let's create a binary classification dataset.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=5, n_informative=2)
X = pd.DataFrame(X)

# Categorical data
X[5] = np.random.choice(["red", "green", "blue"], size=1000)
```

Import and fit the encoder:

```python
import bayte as bt

encoder = bt.BayesianTargetEncoder(dist="bernoulli")
encoder.fit(X[[5]], y)
```

To encode your categorical data,

```python
X[5] = encoder.transform(X[[5]])
```

### Ensemble

If you want to utilize the ensemble methodology described above, construct the same dataset

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=5, n_informative=2)
X = pd.DataFrame(X)

# Categorical data
X[5] = np.random.choice(["red", "green", "blue"], size=1000)
```

and import a classifier to supply to the ensemble class

```python
from sklearn.svm import SVC

import bayte as bt

ensemble = bt.BayesianTargetClassifier(
    base_estimator=SVC(kernel="linear"),
    encoder=bt.BayesianTargetEncoder(dist="bernoulli")
)
```

Fit the ensemble. **NOTE**: either supply an explicit list of categorical features to `categorical_feature`, or
use a DataFrame with categorical data types.

```python
ensemble.fit(X, y, categorical_feature=[5])
```

When you call ``predict`` on a novel dataset, note that the encoder will transform your data at runtime and it
will encode based on the *mean of the posterior distribution*:

```python
ensemble.predict(X)
```

==================
Encoding algorithm
==================

Refresher on bayesian statistics
--------------------------------

In bayesian statstics, we have

.. math::
    p(\theta | y) = \frac{p(y | \theta)p(\theta)}{p(y)}

where :math:`p(\theta)` is the *prior distribution* for parameter :math:`\theta`,
:math:`p(y|\theta)` is the *likelihood* of :math:`y` given :math:`\theta`, and
:math:`p(\theta|y)` is the *posterior distribution* of parameter :math:`\theta`
using :math:`y`. In particular, we will focus on *conjugate Bayesian models*,
where the prior distribution and posterior distribution of :math:`\theta` are
from the same family.

Example
~~~~~~~

Consider a situation where the target variable in our dataset is binary. This
means that :math:`y_{1}, ..., y_{n}` are independent and identically distributed
from a Bernoulli process where :math:`\theta`, the probability of a 1, is **unknown**.

Using `Fink's Compendium of conjugate priors <https://www.johndcook.com/CompendiumOfConjugatePriors.pdf>`_,
the prior distribution of :math:`\theta` is a Beta distribution with **hyperparameters**
:math:`\alpha` and :math:`\beta`. i.e., :math:`\theta \sim Beta(\alpha, \beta)`

.. math:
    
    p(\theta) = p(\theta|\alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)}p^{\alpha - 1}(1 - p)^{\beta - 1}, 0 < p < 1

Since we are using a conjugate Bayesian model, the posterior distribution :math:`p(\theta | y)`
follows a :math:`Beta(\alpha^{\prime}, \beta^{\prime})`. Fink stipulates that 

.. math::

    \alpha^{\prime} = \alpha + \sum_{i = 1}^{n} y_{i}

and

.. math::

    \beta^{\prime} = \beta + n - \sum_{i = 1}^{n} y_{i}

Procedure
---------

Ok, let's lay out the procedure for bayesian target encoding. Suppose you have :math:`n`
training observations, with :math:`Y = (y_{1}, ..., y_{n})` representing the target and
categorical variable :math:`X_{1} = (x_{1}, ..., x_{n})` with distinct values
:math:`V = (v_{1}, ..., v_{l})`.

#. Choose a likelihood for the target variable (e.g. Bernoulli for binary classification),
#. Derive the conjugate prior for the likelihood (e.g. Beta),
#. Use the training data to initialize the hyperparameters for the prior distribution (e.g. :math:`\alpha` and :math:`\beta`) [1]_,
#. Derive the methodology for generating the posterior distribution parameters,
#. For each level :math:`v_{i} \in V`,

   #. Generate the posterior distribution using :math:`y_{1}, ..., y_{m} | x_{j} = v_{i}, \forall j \in (1, m)`,
   #. Set the encoding value to a sample from the posterior distribution [2]_

.. [1] Initializing the hyperparameters is generally reliant on common interpretations.
.. [2] If a new level has appeared in the dataset, the encoding will be sampled from the prior distribution.

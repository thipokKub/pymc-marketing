import pymc as pm
import numpy as np
import pytensor.tensor as pt

from pymc_marketing.mmm.components.base import ComponentBase

# To change output node likelihood
# [x] - Normal
# [x] - Student-t
# [ ] - Negative binomial (require link function)

class LikelihoodComponent(ComponentBase):
    def __init__(self, name, t):
        super().__init__(name)

class NormalLikelihood(LikelihoodComponent):
    _name = "Likelihood.Normal"
    def __init__(
        self, trend, seasonal, media, control, observed, **kwargs
    ):
        super().__init__(self._name)
        self._sigma = pm.HalfNormal("sigma", sigma=1)
        self._mu = pm.Deterministic("mu", var=(
            trend + seasonal + media + control
        ))
        self._likelihood = pm.Normal("likelihood", mu=self._mu, sigma=self._sigma, observed=observed)

class StudentTLikelihood(LikelihoodComponent):
    _name = "Likelihood.StudentT"
    def __init__(
        self, trend, seasonal, media, control, observed, **kwargs
    ):
        super().__init__(self._name)
        nu = pm.Gamma("nu", alpha=25, beta=2)
        self._dof = pm.Deterministic("dof", var=(nu + 1))
        self._mu = pm.Deterministic("mu", var=(
            trend + seasonal + media + control
        ))
        self._sigma = pm.HalfNormal("sigma", sigma=1)
        self._likelihood = pm.StudentT(
            "likelihood",
            nu=self._dof,
            mu=self._mu,
            sigma=self._sigma,
            observed=observed
        )

class NegativeBinomialLikelihood(LikelihoodComponent):
    _name = "Likelihood.NegativeBinomial"
    _name = "Likelihood.StudentT"
    def __init__(
        self, trend, seasonal, media, control, observed, **kwargs
    ):
        super().__init__(self._name)
        # TODO: Implement
        raise NotImplementedError
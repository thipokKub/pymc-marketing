import pymc as pm
import numpy as np
import pytensor.tensor as pt

from pymc_marketing.mmm.components.base import ComponentBase
from pymc_marketing.mmm.components.custom_distributions import FoldedNormal

# To change output node likelihood
# - Off load `mu` calculation to the main model. In case of additional output (multi-outputs)
# [x] - Normal
# [x] - Student-t
# [x] - Truncated Normal (Zero truncated) - Always >= 0
# [x] - Truncated Student-t (Zero truncated) - Always >= 0
# [x] - Folded Normal - Always >= 0
# [?] - *BiTApprox: Depend on BiNormal - SkewT doesn't have close form, but it can be approximated by pm.Mixture interpolation
class LikelihoodComponent(ComponentBase):
    def __init__(self, name, t):
        super().__init__(name)

class NormalLikelihood(LikelihoodComponent):
    _name = "Likelihood.Normal"
    def __init__(
        self, x, observed, **kwargs
    ):
        name = getattr(kwargs, "name", "")
        super().__init__(f"{self._name}_{name}" if len(name) > 0 else self._name)

        self._sigma = pm.HalfNormal("sigma", sigma=1)
        self._mu = pm.Deterministic("mu", var=x)
        self._likelihood = pm.Normal("likelihood", mu=self._mu, sigma=self._sigma, observed=observed)

class StudentTLikelihood(LikelihoodComponent):
    _name = "Likelihood.StudentT"
    def __init__(
        self, x, observed, **kwargs
    ):
        name = getattr(kwargs, "name", "")
        super().__init__(f"{self._name}_{name}" if len(name) > 0 else self._name)
        
        nu = pm.Gamma("nu", alpha=25, beta=2)
        self._dof = pm.Deterministic("dof", var=(nu + 1))
        self._mu = pm.Deterministic("mu", var=x)
        self._sigma = pm.HalfNormal("sigma", sigma=1)
        self._likelihood = pm.StudentT(
            "likelihood",
            nu=self._dof,
            mu=self._mu,
            sigma=self._sigma,
            observed=observed
        )

class TruncatedNormalLikelihood(LikelihoodComponent):
    _name = "Likelihood.TruncatedNormal"
    def __init__(
        self, x, observed, lower: float = 0, upper: float = None, **kwargs
    ):
        name = getattr(kwargs, "name", "")
        super().__init__(f"{self._name}_{name}" if len(name) > 0 else self._name)

        self._sigma = pm.HalfNormal("sigma", sigma=1)
        self._mu = pm.Deterministic("mu", var=x)
        self._likelihood = pm.TruncatedNormal(
            "likelihood",
            mu=self._mu,
            sigma=self._sigma,
            observed=observed,
            lower=lower,
            upper=upper
        )

class TruncatedStudentTLikelihood(LikelihoodComponent):
    _name = "Likelihood.TruncatedStudentT"
    def __init__(
        self, x, observed, lower: float = 0, upper: float = None, **kwargs
    ):
        name = getattr(kwargs, "name", "")
        super().__init__(f"{self._name}_{name}" if len(name) > 0 else self._name)
        
        nu = pm.Gamma("nu", alpha=25, beta=2)
        self._dof = pm.Deterministic("dof", var=(nu + 1))
        self._mu = pm.Deterministic("mu", var=x)
        self._sigma = pm.HalfNormal("sigma", sigma=1)
        self._likelihood = pm.Truncated(
            "likelihood",
            pm.StudentT.dist(
                nu=self._dof,
                mu=self._mu,
                sigma=self._sigma
            ),
            observed=observed,
            lower=lower,
            upper=upper
        )

class FoldedNormalLikelihood(LikelihoodComponent):
    _name = "Likelihood.FoldedNormal"
    def __init__(
        self, x, observed, lower: float = 0, upper: float = None, **kwargs
    ):
        name = getattr(kwargs, "name", "")
        super().__init__(f"{self._name}_{name}" if len(name) > 0 else self._name)

        self._sigma = pm.HalfNormal("sigma", sigma=1)
        self._mu = pm.Deterministic("mu", var=pt.abs(x)) # Ensure positive
        self._likelihood = FoldedNormal(
            "likelihood",
            mu=self._mu,
            sigma=self._sigma,
            observed=observed
        )

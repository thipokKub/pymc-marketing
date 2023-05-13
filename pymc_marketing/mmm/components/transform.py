import pymc as pm
import numpy as np
import pytensor.tensor as pt

from pymc_marketing.mmm.components.base import ComponentBase
import pymc_marketing.mmm.transformers as transformers

# Wrapper class for media transformations
# [x] - Add Geometric Transform
# [x] - Add Delayed Adstock Transform
# [x] - Add Logistic Saturation Transform
# [x] - Add Tanh Saturation Transform
# [x] - Add Hill Saturation Transform

class TransformComponent(ComponentBase):
    def __init__(self, name, shape):
        super().__init__(name)
        self._shape = tuple(shape)

# Adstock Effect
class GeometricTransform(TransformComponent):
    _name = "Transform.Geometric"
    def __init__(self, x, shape, **kwargs):
        super().__init__(self._name, shape)
        adstock_max_lag: int = getattr(kwargs, "adstock_max_lag", 12)

        self._alpha = pm.Beta(name="alpha", alpha=1, beta=3, shape=self._shape)
        self._adstock = pm.Deterministic("adstock", var=(
            transformers.geometric_adstock(
                x=x,
                alpha=self._alpha,
                l_max=adstock_max_lag,
                normalize=True,
                axis=0
            )
        ))

class DelayedAdstockTransform(TransformComponent):
    _name = "Transform.DelayedAdstock"
    def __init__(self, x, shape, **kwargs):
        super().__init__(self._name, shape)
        adstock_max_lag: int = getattr(kwargs, "adstock_max_lag", 12)

        self._alpha = pm.Beta(name="alpha", alpha=1, beta=3, shape=self._shape)
        self._theta = pm.Truncated(
            "theta", 
            dist=pm.Exponential.dist(lam=1, shape=self._shape),
            lower=0, upper=(adstock_max_lag - 1)
        )
        self._adstock = pm.Deterministic("adstock", var=(
            transformers.delayed_adstock(
                x=x,
                alpha=self._alpha,
                theta=self._theta,
                l_max=adstock_max_lag,
                normalize=True,
                axis=0
            )
        ))

# Shape Effect
class LogisticTransform(TransformComponent):
    _name = "Transform.Logistic"
    def __init__(self, x, shape, **kwargs):
        super().__init__(self._name, shape)

        self._lam = pm.Gamma(name="lam", alpha=3, beta=1, shape=self._shape)
        self._saturate = pm.Deterministic("saturate", var=(
            transformers.logistic_saturation(x, lam=self._lam)
        ))

class TanhTransform(TransformComponent):
    _name = "Transform.Tanh"
    def __init__(self, x, shape, **kwargs):
        super().__init__(self._name, shape)

        eps: int = np.abs(getattr(kwargs, "eps", 1e-4))
        assert eps > 0

        self._b = pt.clip(pm.Gamma("b", alpha=1.5, beta=1, shape=self._shape), eps, np.inf)
        # Must be non-zero?
        # I mean, if c < 0 then it becomes monotonically decreasing function
        # Which I don't think that is how saturation function should work
        # I am sticking with c > 0
        self._c = pt.clip(pm.Gamma("c", alpha=1.5, beta=1, shape=self._shape), eps, np.inf)

        self._saturate = pm.Deterministic("saturate", var=(
            transformers.tanh_saturation(x, b=self._b, c=self._c)
        ))

class HillTransform(TransformComponent):
    _name = "Transform.Hill"
    def __init__(self, x, shape, **kwargs):
        super().__init__(self._name, shape)

        eps: int = np.abs(getattr(kwargs, "eps", 1e-4))
        assert eps > 0

        self._k = pt.clip(pm.Gamma("k", alpha=1, beta=1, shape=self._shape), eps, np.inf)
        self._s = pt.clip(pm.Gamma("s", alpha=1, beta=1, shape=self._shape), eps, np.inf)
        
        self._saturate = pm.Deterministic("saturate", var=(
            pt.where(x > 0, 1/(1 + (pt.where(x < eps, eps, x)/self._k)**(-self._s)), 0)
        ))
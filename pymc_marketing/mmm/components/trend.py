import pymc as pm
import numpy as np
import pytensor.tensor as pt

from pymc_marketing.mmm.components.base import ComponentBase
from pymc_marketing.mmm.components.utils import Spline

# TODO
# [x] - Add Linear: a*t + b
# [x] - Add Logarithmic: a*log(1 + t) + b
# [x] - Add Power: a*t^k + b
# [x] - Add Exponential: a*exp(k*t) + b
# [x] - Add Polynomial: a_0 + a_1 * t + a_2 * t^2 + ... + a_n * t^n
# [x] - Add Logistic: L/(1 + exp(-k*(x - c))) + b
# [x] - Add Spline/Piecewise
# [?] - Need to define prediction behaviour
#       Want to include damped trend, but not sure how to train that

class TrendComponent(ComponentBase):
    def __init__(self, name, t):
        super().__init__(name)
        self._t = pm.MutableData("t", t)

class LinearTrend(TrendComponent):
    _name = "Trend.Linear"
    def __init__(self, t, **kwargs):
        super().__init__(self._name, t)
        zero_slope = getattr(kwargs, "zero_slope", False)
        zero_intercept = getattr(kwargs, "zero_intercept", False)
        
        self._slope = 0 if zero_slope else pm.Normal("slope", mu=0, sigma=1)
        self._intercept = 0 if zero_intercept else pm.Normal("intercept", mu=0, sigma=1)
        self._trend = pm.Deterministic("trend", var=(self._slope * self._t + self._intercept))

class LogarithmicTrend(TrendComponent):
    _name = "Trend.Logarithmic"
    def __init__(self, t, **kwargs):
        super().__init__(self._name, t)
        zero_slope = getattr(kwargs, "zero_slope", False)
        zero_intercept = getattr(kwargs, "zero_intercept", False)

        self._slope = 0 if zero_slope else pm.Normal("slope", mu=0, sigma=1)
        self._intercept = 0 if zero_intercept else pm.Normal("intercept", mu=0, sigma=1)
        self._trend = pm.Deterministic("trend", var=(self._slope * pt.log(self._t + 1) + self._intercept))

class PowerTrend(TrendComponent):
    _name = "Trend.Power"
    def __init__(self, t, **kwargs):
        super().__init__(self._name, t)
        n_order = getattr(kwargs, "n_order", 1)
        zero_slope = getattr(kwargs, "zero_slope", False)
        zero_intercept = getattr(kwargs, "zero_intercept", False)

        self._slope = 0 if zero_slope else pm.Normal("slope", mu=0, sigma=1)
        self._intercept = 0 if zero_intercept else pm.Normal("intercept", mu=0, sigma=1)
        self._trend = pm.Deterministic("trend", var=(self._slope * pt.power(self._t, n_order) + self._intercept))

class ExponentialTrend(TrendComponent):
    _name = "Trend.Exponential"
    def __init__(self, t, **kwargs):
        super().__init__(self._name, t)
        zero_slope = getattr(kwargs, "zero_slope", False)
        zero_intercept = getattr(kwargs, "zero_intercept", False)

        self._slope = 0 if zero_slope else pm.Normal("slope", mu=0, sigma=1)
        self._intercept = 0 if zero_intercept else pm.Normal("intercept", mu=0, sigma=1)
        self._rate = pm.Normal("rate", mu=0, sigma=1)
        self._trend = pm.Deterministic("trend", var=(self._slope * pt.exp(self._rate * self._t) + self._intercept))

class PolynomialTrend(TrendComponent):
    _name = "Trend.Polynomial"
    def __init__(self, t, **kwargs):
        super().__init__(self._name, t)
        n_order = getattr(kwargs, "n_order", 1)
        assert n_order >= 0

        self._slope = pm.Normal("slope", mu=0, sigma=1, shape=(n_order + 1, ))
        self._trend = pm.Deterministic("trend", var=(
            (self._slope * pt.stack([pt.power(self._t, i) for i in range(n_order + 1)], axis=1)).sum(axis=1)
        ))
        
class LogisticTrend(TrendComponent):
    _name = "Trend.Logistic"
    def __init__(self, t, **kwargs):
        super().__init__(self._name, t)
        
        self._max_L = pm.HalfNormal("max_L", sigma=1)
        self._growth_rate = pm.Normal("growth_rate", mu=0, sigma=1)
        self._x_offset = pm.Normal("x_offset", mu=0, sigma=1)
        self._y_offset = pm.Normal("y_offset", mu=0, sigma=1)
        
        self._trend = pm.Deterministic("trend", var=(
            self._max_L/(1 + pm.math.exp(-self._growth_rate * (self._t - self._x_offset))) + self._y_offset
        ))

# Haven't tested
class SplineTrend(TrendComponent):
    _name = "Trend.Spline"
    def __init__(
            self,
            t,
            checkpoints,
            spline: Spline.SplineType = Spline.SplineType.LINEAR,
            **kwargs
        ):
        super().__init__(self._name, t)
        self._checkpoints = checkpoints
        if type(checkpoints) is int:
            self._checkpoints = np.linspace(t.min(), t.max(), checkpoints + 2)[1:-1]
        spline_kernel = Spline.get_spline(t, self._checkpoints, spline, **kwargs)
        
        self._sigma = pm.HalfNormal("sigma", sigma=1)
        self._anchors = pt.cumsum(pm.Laplace("anchors", mu=0, b=self._sigma, shape=(len(self._checkpoints), )))
        self._trend = pm.Deterministic("trend", var=(pm.math.dot(spline_kernel, self._anchors)))
    
    def _optional_coords(self):
        return ({
            f"{self._name}.spline": np.arange(len(self._checkpoints))
        })
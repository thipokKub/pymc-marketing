import pymc as pm
import numpy as np
import pytensor.tensor as pt

from pymc_marketing.mmm.components.base import ComponentBase
from pymc_marketing.mmm.components.utils import Spline

# TODO
# [x] - Add Linear: a*t + b
# [ ] - Add Logarithmic: a*log(1 + t) + b
# [ ] - Add Power: a*t^k + b
# [ ] - Add Exponential: a*exp(k*t) + b
# [ ] - Add Polynomial: a_0 + a_1 * t + a_2 * t^2 + ... + a_n * t^n
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

class SplineTrend(TrendComponent):
    _name = "Trend.Spline"
    def __init__(self, t, checkpoints, spline: Spline.SplineType = Spline.SplineType.LINEAR, **kwargs):
        super().__init__(self._name, t)
        if type(checkpoints) is int:
            checkpoints = np.linspace(t.min(), t.max(), checkpoints + 2)[1:-1]
        spline_kernel = Spline.get_spline(t, checkpoints, spline, **kwargs)
        
        self._sigma = pm.HalfNormal("sigma", sigma=1)
        self._anchors = pt.cumsum(pm.Laplace("anchors", mu=0, b=self._sigma, shape=(len(checkpoints), )))
        self._trend = pm.Deterministic("trend", var=(pm.math.dot(spline_kernel, self._anchors)))
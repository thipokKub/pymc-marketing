import pymc as pm
import numpy as np
import pytensor.tensor as pt

from pymc_marketing.mmm.components.base import ComponentBase
from pymc_marketing.mmm.components.utils import Spline

class SeasonalityComponent(ComponentBase):
    def __init__(self, name, fourier_features):
        super().__init__(name)
        self._fourier_features = pm.MutableData("fourier_features", fourier_features)

class ConstantSeasonality(SeasonalityComponent):
    _name = "Seasonality.Constant"
    def __init__(self, fourier_features, **kwargs):
        super().__init__(self._name, fourier_features)
        size = fourier_features.shape[1]
        self._coeff = pm.Normal("coeff", mu=0, sigma=1, shape=(size,))
        self._seasonality = pm.Deterministic("seasonality", var=(self._coeff * self._fourier_features).sum(axis=1))

class SplineSeasonality(SeasonalityComponent):
    _name = "Seasonality.Spline"
    def __init__(
        self,
        fourier_features,
        t, checkpoints,
        spline: Spline.SplineType = Spline.SplineType.LINEAR,
        **kwargs
    ):
        super().__init__(self._name, fourier_features)
        if type(checkpoints) is int:
            self._checkpoints = np.linspace(t.min(), t.max(), checkpoints + 2)[1:-1]
        spline_kernel = Spline.get_spline(t, self._checkpoint, spline, **kwargs)
        if getattr(kwargs, "dim", None) is not None:
            dim_kwargs = {"dim": [
                *self._optional_coords().keys(),
                getattr(kwargs, "dim", None)
            ]}
        else:
            dim_kwargs = {"shape": (len(self._checkpoint), fourier_features.shape[1],)}
        
        self._sigma = pm.HalfNormal("sigma", sigma=1)
        self._anchors = pt.cumsum(pm.Laplace("anchors", mu=0, b=self._sigma, **dim_kwargs), axis=0)
        self._seasonality = pm.Deterministic(
            "seasonality", var=(self._fourier_features * pm.math.dot(spline_kernel, self._anchors)).sum(axis=1)
        )

    def _optional_coords(self):
        return ({
            **super()._optional_coords(),
            **{
                f"{self._name}.spline": np.arange(len(self._checkpoints))
            }
        })
import pymc as pm
import numpy as np
import pytensor.tensor as pt

from pymc_marketing.mmm.components.base import ComponentBase
from pymc_marketing.mmm.components.utils import Spline

# Assume that media variable had been transformed
# TODO
# [ ] - Add Constant
# [ ] - Add KTR
# [ ] - Add Gaussian Random Walk
# [ ] - Add GPs
# [ ] - Add Hierarchical Constant
# [ ] - Add Hierarchical GPs
class MediaComponent(ComponentBase):
    def __init__(self, name, t, media):
        super().__init__(name)
        self._t = pm.MutableData("t", t) 
        self._media = pm.MutableData("media", media)

# class ConstantMedia(ComponentBase):

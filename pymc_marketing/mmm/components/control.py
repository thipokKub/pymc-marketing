import pymc as pm
import numpy as np
import pytensor.tensor as pt

from pymc_marketing.mmm.components.base import ComponentBase

# TODO
# [ ] - Add constant
# I'm not sure about time-varying coefficient here since we want to stratify the outcome (right?)

class ControlComponent(ComponentBase):
    def __init__(self, name, t, control):
        super().__init__(name)
        self._t = pm.MutableData("t", t) 
        self._control = pm.MutableData("control", control)

class ConstantControl(ControlComponent):
    _name = "Control.Constant"
    def __init__(self, t, control, **kwargs):
        super().__init__(self._name, t, control)
        
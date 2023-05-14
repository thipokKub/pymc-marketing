# Implement Non-Official Distributions in PyMC

import numpy as np
from scipy import stats
import pytensor.tensor as pt
from pymc.pytensorf import floatX
from typing import List, Optional, Union
from pytensor.tensor.random.op import RandomVariable
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.continuous import PositiveContinuous

# Implement FoldedNormal Distribution
class FoldedNormalRV(RandomVariable):
    name = "folded_normal"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("FoldedNormal", "\\operatorname{FoldedNormal}")
    
    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState,
        mu: Union[np.ndarray, float],
        sigma: Union[np.ndarray, float],
        size: Optional[Union[List[int], int]],
    ) -> np.ndarray:
        return stats.foldnorm.rvs(
            mu/sigma,
            loc=0,
            scale=sigma,
            size=size,
            random_state=rng,
        )

folded_normal = FoldedNormalRV()

class FoldedNormal(PositiveContinuous):
    rv_op = folded_normal
    
    @classmethod
    def dist(cls, mu=0, sigma=None, *args, **kwargs):
        mu = pt.as_tensor_variable(floatX(mu))
        sigma = pt.as_tensor_variable(floatX(sigma))
        return super().dist([mu, sigma], *args, **kwargs)
    
    def moment(rv, size, mu, sigma):
        mean = ((2/np.pi)**0.5) * sigma * pt.exp(-(mu**2)/(2*(sigma**2))) - mu * pt.erf(-mu/(sigma * (2**0.5)))
        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean
    
    def logp(value, mu, sigma):
        res = (
            -1 * pt.log(sigma)
            + 1/2 * (pt.log(2) - pt.log(np.pi))
            - ((value**2 + mu**2)/(2*sigma**2))
            + pt.log(pt.cosh(mu*value/(sigma**2)))
        )
        res = pt.switch(pt.gt(value, 0.0), res, -np.inf)
        return check_parameters(
            res,
            sigma > 0,
            mu >= 0,
            msg="mu >= 0, sigma > 0",
        )
    
    def logcdf(value, mu, sigma):
        res = pt.switch(
            pt.le(value, 0),
            -np.inf,
            pt.log(
                pt.erf((value + mu)/(sigma * (2**0.5)))
                + pt.erf((value - mu)/(sigma * (2**0.5)))
            ) - pt.log(2)
        )
        
        return check_parameters(
            res,
            sigma > 0,
            mu >= 0,
            msg="mu >= 0, sigma > 0",
        )
    

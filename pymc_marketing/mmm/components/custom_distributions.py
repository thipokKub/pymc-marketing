# Implement Non-Official Distributions in PyMC

import pymc as pm
import numpy as np
from scipy import stats
import pytensor.tensor as pt
from pymc.pytensorf import floatX
from typing import List, Optional, Union
from pytensor.tensor.random.op import RandomVariable
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.shape_utils import rv_size_is_none
from scipy.stats._distn_infrastructure import rv_continuous, _ShapeInfo
from pymc.distributions.continuous import PositiveContinuous, Continuous

# TODO:
# [x] - FoldedNormal Distribution
# [x] - FoldedCauchy Distribution: Need to de-cipher `c` constant first
#       (see https://docs.scipy.org/doc/scipy/tutorial/stats/continuous_foldcauchy.html)
# [x] - *FoldedStudentT Distribution: Maybe? https://en.wikipedia.org/wiki/Folded-t_and_half-t_distributions
#       Need to write custom rv_continuous. But based on `stats.foldcauchy``, this should be doable
#       https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.foldcauchy.html#scipy.stats.foldcauchy
# [x] - BiCauchy, or SkewCauchy
# [?] - *BiNormal, or SkewNormal: Need to find cdf formula first
#       Need to write custom rv_continuous, not sure how
# [?] - *BiTApprox: Depend on BiNormal - SkewT doesn't have close form, but it can be approximated by pm.Mixture interpolation
#       Need to write custom rv_continuous, not sure how
# [x] - Symmetric Generalized Normal Distribution
#       https://en.wikipedia.org/wiki/Generalized_normal_distribution
# [-] - *Asymmetric Generalized Normal Distribution
#       Need to write custom rv_continuous 

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
    def dist(cls, mu=0.0, sigma=1.0, *args, **kwargs):
        mu = pt.as_tensor_variable(floatX(mu))
        sigma = pt.as_tensor_variable(floatX(sigma))
        return super().dist([mu, sigma], *args, **kwargs)
    
    def moment(rv, size, mu, sigma):
        mean = (
            ((2/np.pi)**0.5) * sigma * pt.exp(-(mu**2)/(2*(sigma**2)))
            - mu * pt.erf(-mu/(sigma * (2**0.5)))
        )
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
    
# Implement Folded-Cauchy Distribution
class FoldedCauchy(RandomVariable):
    name = "folded_cauchy"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "floatX"
    _print_name = ("FoldedCauchy", "\\operatorname{FoldedCauchy}")
    
    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState,
        mu: Union[np.ndarray, float],
        sigma: Union[np.ndarray, float],
        size: Optional[Union[List[int], int]],
    ) -> np.ndarray:
        return stats.foldcauchy.rvs(
            c=mu/sigma,
            loc=0,
            scale=sigma,
            size=size,
            random_state=rng,
        )
    
foldedCauchy = FoldedCauchy()

class FoldedCauchy(PositiveContinuous):
    rv_op = foldedCauchy
    
    @classmethod
    def dist(cls, mu=0.0, sigma=1.0, *args, **kwargs):
        mu = pt.as_tensor_variable(floatX(mu))
        sigma = pt.as_tensor_variable(floatX(sigma))
        return super().dist([mu, sigma], *args, **kwargs)
    
    def moment(rv, size, mu, sigma):
        mu, _ = pt.broadcast_arrays(mu, sigma)
        if not rv_size_is_none(size):
            mu = pt.full(size, mu)
        return mu
    
    def logp(value, mu, sigma):
        c = mu/sigma
        res = (
            pt.log(
                (1.0/np.pi) * 1.0/(1 + (value/sigma - c)**2) +
                (1.0/np.pi) * 1.0/(1 + (value/sigma + c)**2)
            )
            - pt.log(sigma)
        )
        res = pt.switch(pt.gt(value/sigma, 0), res, -np.inf)
        return check_parameters(
            res,
            sigma > 0,
            mu >= 0,
            msg="mu >= 0, sigma > 0",
        )
    
    def logcdf(value, mu, sigma):
        c = mu/sigma
        res = pt.switch(
            pt.le(value/sigma, 0),
            -np.inf,
            (
                -1 * pt.log(np.pi)
                + pt.log(
                    pt.arctan(value/sigma - c)
                    + pt.arctan(value/sigma + c)
                )
            )
        )
        return check_parameters(
            res,
            sigma > 0,
            mu >= 0,
            msg="mu >= 0, sigma > 0",
        )
    
# Implement Folded-StudentT
class foldt_gen(rv_continuous):
    def _argcheck(self, c, df):
        return c >= 0 and df > 0

    def _shape_info(self):
        return [
            _ShapeInfo("c", False, (0, np.inf), (True, False)),
            _ShapeInfo("df", False, (0, np.inf), (False, False))
        ]

    def _rvs(self, c, df, size=1, random_state=None):
        return abs(stats.t.rvs(df, loc=c, size=size, random_state=random_state))
    
    def _pdf(self, x, c, df):
        return np.where(x >=0, stats.t.pdf(x, df, loc=c) + stats.t.pdf(x, df, loc=-c), 0)

    def _cdf(self, x, c, df):
        return np.where(x <= 0, 0, stats.t.cdf(x, df, loc=c) + stats.t.cdf(x, df, loc=-c))

    def _stats(self, c, df):
        return np.inf, np.inf, np.nan, np.nan
    
fold_t = t = foldt_gen(name='fold_t')

class FoldedStudentTRV(RandomVariable):
    name = "FoldedStudentT"
    ndim_supp = 0
    ndims_params = [0, 0, 0]
    dtype = "floatX"
    _print_name = ("FoldedStudentT", "\\operatorname{FoldedStudentT}")
    
    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState,
        nu: Union[np.ndarray, float],
        mu: Union[np.ndarray, float],
        sigma: Union[np.ndarray, float],
        size: Optional[Union[List[int], int]],
    ) -> np.ndarray:
        return fold_t.rvs(
            c=mu/sigma,
            df=nu,
            loc=0,
            scale=sigma,
            size=size,
            random_state=rng,
        )
    
foldedStudentT = FoldedStudentTRV()

class FoldedStudentT(PositiveContinuous):
    rv_op = foldedStudentT
    
    @classmethod
    def dist(cls, nu=1.0, mu=0.0, sigma=1.0, *args, **kwargs):
        nu = pt.as_tensor_variable(floatX(nu))
        mu = pt.as_tensor_variable(floatX(mu))
        sigma = pt.as_tensor_variable(floatX(sigma))
        return super().dist([nu, mu, sigma], *args, **kwargs)
    
    def moment(rv, size, nu, mu, sigma):
        _, mu, sigma = pt.broadcast_arrays(nu, mu, sigma)
        mode = mu/sigma
        if not rv_size_is_none(size):
            mode = pt.full(size, mode)
        return mode
    
    def logp(value, nu, mu, sigma):
        c = mu/sigma
        res = (
            pt.log(
                pt.exp(pm.StudentT.logp(value/sigma, nu, c, 1)) +
                pt.exp(pm.StudentT.logp(value/sigma, nu, -c, 1))
            )
            - pt.log(sigma)
        )
        res = pt.switch(pt.gt(value/sigma, 0), res, -np.inf)
        return check_parameters(
            res,
            nu > 0,
            mu >= 0,
            sigma > 0,
            msg="nu > 0, mu >= 0, sigma > 0",
        )
    
    def logcdf(value, nu, mu, sigma):
        c = mu/sigma
        res = pt.log(
            pt.exp(pm.StudentT.logcdf(value/sigma, nu, c, 1)) +
            pt.exp(pm.StudentT.logcdf(value/sigma, nu, -c, 1))
        )
        pt.switch(pt.le(value/sigma, 0), res)
        return check_parameters(
            res,
            nu > 0,
            mu >= 0,
            sigma > 0,
            msg="nu > 0, mu >= 0, sigma > 0",
        )

# Implement Bi-Cauchy Distribution
class BiCauchyRV(RandomVariable):
    name = "bi_cauchy"
    ndim_supp = 0
    ndims_params = [0, 0, 0]
    dtype = "floatX"
    _print_name = ("BiCauchy", "\\operatorname{BiCauchy}")
    
    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState,
        mu: Union[np.ndarray, float],
        sigma: Union[np.ndarray, float],
        alpha: Union[np.ndarray, float],
        size: Optional[Union[List[int], int]],
    ) -> np.ndarray:
        return stats.skewcauchy.rvs(
            a=alpha,
            loc=mu,
            scale=sigma,
            size=size,
            random_state=rng,
        )

bicauchy = BiCauchyRV()
class BiCauchy(Continuous):
    """
    Renamed to BiCauchy, to support the same family as BiGaussian
    See https://en.wikipedia.org/wiki/Skewed_generalized_t_distribution#Skewed_Cauchy_distribution
    
    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mode      :math:`\mu`
    Mean      undefined
    Variance  undefined
    ========  ========================
    """
    
    rv_op = bicauchy
    
    @classmethod
    def dist(cls, mu=0.0, sigma=1.0, alpha=0.0, *args, **kwargs):
        alpha = pt.as_tensor_variable(floatX(alpha))
        mu = pt.as_tensor_variable(floatX(mu))
        sigma = pt.as_tensor_variable(floatX(sigma))
        return super().dist([mu, sigma, alpha], *args, **kwargs)
    
    def moment(rv, size, mu, sigma, alpha):
        mu, _, _ = pt.broadcast_arrays(mu, sigma, alpha)
        if not rv_size_is_none(size):
            mu = pt.full(size, mu)
        return mu
    
    def logp(value, mu, sigma, alpha):
        res = -1 * (
            pt.log(sigma)
            + pt.log(np.pi)
            + pt.log(1 + (value - mu)**2 / (sigma**2 * (1 + alpha * pt.sign(value - mu))**2) )
        )
        return check_parameters(
            res,
            sigma > 0,
            pt.abs(alpha) < 1,
            msg="sigma > 0, -1 < alpha < 1",
        )
    
    def logcdf(value, mu, sigma, alpha):
        res = pt.log(
            pt.switch(
                pt.le(value, 0),
                (1 - alpha)/2 + ((1 - alpha)/np.pi) * pt.arctan((value - mu)/(sigma * (1 - alpha))),
                (1 - alpha)/2 + ((1 + alpha)/np.pi) * pt.arctan((value - mu)/(sigma * (1 + alpha)))
            )
        )
        return check_parameters(
            res,
            sigma > 0,
            pt.abs(alpha) < 1,
            msg="sigma > 0, -1 < alpha < 1",
        )

# Implement Generalized Normal Distribution
class GenNormalRV(RandomVariable):
    name = "genalized_normal"
    ndim_supp = 0
    ndims_params = [0, 0, 0]
    dtype = "floatX"
    _print_name = ("GenNormal", "\\operatorname{GenNormal}")
    
    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState,
        mu: Union[np.ndarray, float],
        sigma: Union[np.ndarray, float],
        beta: Union[np.ndarray, float],
        size: Optional[Union[List[int], int]],
    ) -> np.ndarray:
        return stats.gennorm.rvs(
            beta=beta,
            loc=mu,
            scale=sigma,
            size=size,
            random_state=rng,
        )
    
gennorm = GenNormalRV()

class GenNormal(Continuous):
    """
    See https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1
    
    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mode      :math:`\mu`
    Mean      :math:`\mu`
    Variance  :math:`\mu`
    ========  ========================
    """
    
    rv_op = gennorm
    
    @classmethod
    def dist(cls, mu=0.0, sigma=1.0, beta=1.0, *args, **kwargs):
        beta = pt.as_tensor_variable(floatX(beta))
        mu = pt.as_tensor_variable(floatX(mu))
        sigma = pt.as_tensor_variable(floatX(sigma))
        return super().dist([mu, sigma, beta], *args, **kwargs)
    
    def moment(rv, size, mu, sigma, alpha):
        mu, _, _ = pt.broadcast_arrays(mu, sigma, alpha)
        if not rv_size_is_none(size):
            mu = pt.full(size, mu)
        return mu
    
    def logp(value, mu, sigma, beta):
        res = (
            pt.log(0.5 * beta)
            - pt.gammaln(1.0/beta)
            - pt.abs((value - mu)/sigma)**beta
            - pt.log(sigma)
        )
        return check_parameters(
            res,
            sigma > 0,
            beta > 0,
            msg="sigma > 0, beta > 0"
        )
    
    def logcdf(value, mu, sigma, beta):
        c = 0.5 * pt.sign(value)
        res = pt.log(
            (0.5 + c)
            - c * pt.gammaincc(1.0/beta, pt.abs((value - mu)/sigma)**beta)
        )
        return check_parameters(
            res,
            sigma > 0,
            beta > 0,
            msg="sigma > 0, beta > 0"
        )

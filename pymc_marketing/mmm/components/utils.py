import numpy as np
import pandas as pd
from enum import Enum

class Spline:
    # Enum
    class SplineType(Enum):
        LINEAR = "Linear"
        GAUSSIAN = "Gaussian"
        PARABOLIC = "Parabolic"
        
    # # Helpers
    @staticmethod
    def get_spline(t, t_refs, ktype: SplineType = SplineType.LINEAR, **kwargs):
        if ktype == Spline.SplineType.LINEAR:
            return Spline.get_linear_spline(t, t_refs, **kwargs)
        elif ktype == Spline.SplineType.GAUSSIAN:
            return Spline.get_gaussian_spline(t, t_refs, **kwargs)
        elif ktype == Spline.SplineType.PARABOLIC:
            return Spline.get_parabolic_spline(t, t_refs, **kwargs)
        else:
            raise NotImplementedError
    
    @staticmethod
    def get_linear_spline(t, t_refs):
        ker = np.zeros((t.shape[0], t_refs.shape[0]))
        ker[t < t_refs[0], 0] = 1
        for idx in range(t_refs.shape[0] - 1):
            valid_idxes = (t >= t_refs[idx]) & (t < t_refs[idx + 1])
            norm = t_refs[idx + 1] - t_refs[idx]
            ker[valid_idxes, idx] = np.abs(t[valid_idxes] - t_refs[idx + 1])/norm
            ker[valid_idxes, idx + 1] = np.abs(t[valid_idxes] - t_refs[idx])/norm
        ker[t >= t_refs[-1], -1] = 1
        return ker / np.sum(ker, axis=1, keepdims=True)
    
    @staticmethod
    def get_gaussian_spline(t, t_refs, rho: float = 0.1, alpha: float = 1, point_to_flatten: float = 1):
        ker = np.where(
            (t <= point_to_flatten).reshape(-1, 1),
            (alpha**2) * np.exp(-1 * np.power(t.reshape(-1, 1) - t_refs.reshape(1, -1), 2)/(2 * rho**2 )),
            (alpha**2) * np.exp(-1 * np.power(np.array([point_to_flatten] * t.shape[0]).reshape(-1, 1) - t_refs.reshape(1, -1), 2)/(2 * rho**2 ))
        )
        return ker / np.sum(ker, axis=1, keepdims=True)
    
    @staticmethod
    def get_parabolic_spline(t, t_refs):
        ker = np.zeros((t.shape[0], t_refs.shape[0]))
        ker[t < t_refs[0], 0] = 1
        for idx in range(t_refs.shape[0] - 1):
            valid_idxes = (t >= t_refs[idx]) & (t < t_refs[idx + 1])
            norm = t_refs[idx + 1] - t_refs[idx]
            ker[valid_idxes, idx] = 0.75 * (1 - ((t[valid_idxes] - t_refs[idx])/norm)**2)
            ker[valid_idxes, idx + 1] = 0.75 *(1 - ((t[valid_idxes] - t_refs[idx + 1])/norm)**2)
        ker[t >= t_refs[-1], -1] = 1
        return ker / np.sum(ker, axis=1, keepdims=True)
    
class Seasonal:
    @staticmethod
    def get_fourier_features(dates, n_order):
        periods = dates.day_of_year / 365.25
        return pd.DataFrame(
            {
                f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods * order)
                for order in range(1, n_order + 1)
                for func in ("sin", "cos")
            }
        )
from __future__ import annotations

from abc import abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd

from .UniqueEncoder import BaseEncoder

# BaseEncoder / NotFittedError come from the previous file


class BaseScaler(BaseEncoder):

    def __init__(self, with_nan: bool = True, clip: bool = False,
                 copy_name: bool = True) -> None:
        super().__init__()
        self.with_nan = with_nan
        self.clip = clip
        self.copy_name = copy_name
        self.n_samples_seen_: int | None = None
        self._clip_range_: Tuple[float, float] | None = None

    # ------------------------------------------------------ hooks (abstract)
    @abstractmethod
    def _learn(self, values: np.ndarray) -> None: ...

    @abstractmethod
    def _apply(self, values: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def _revert(self, values: np.ndarray) -> np.ndarray: ...

    # --------------------------------------------------------------- helpers
    def _to_numeric(self, series) -> pd.Series:
        series = self._validate_series(series)
        if not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors="coerce")
            if series.isna().all():
                raise TypeError("Series could not be converted to numeric.")
        s = series.astype("float64")
        if s.isna().any() and not self.with_nan:
            raise ValueError("NaN found but with_nan=False.")
        return s

    @staticmethod
    def _safe_denominator(value: float, name: str) -> float:
        """Guard against divide-by-zero on constant / degenerate Series."""
        if value == 0 or np.isclose(value, 0.0):
            import warnings
            warnings.warn(
                f"{name} is zero — Series appears constant. "
                "Using 1.0 so all values map to a single point.",
                RuntimeWarning, stacklevel=2,
            )
            return 1.0
        return float(value)

    def _build(self, arr: np.ndarray, src: pd.Series) -> pd.Series:
        return pd.Series(
            arr, index=src.index,
            name=src.name if self.copy_name else None,
            dtype="float64",
        )

    # ------------------------------------------------- template method: fit
    def fit(self, series: pd.Series) -> "BaseScaler":
        s = self._to_numeric(series)
        vals = s.dropna().to_numpy()
        if vals.size == 0:
            raise ValueError("Cannot fit on an all-NaN / empty Series.")

        self.n_samples_seen_ = int(vals.size)
        self._learn(vals)

        if self.clip:
            out = self._apply(vals)
            self._clip_range_ = (float(np.min(out)), float(np.max(out)))

        self._is_fitted = True
        return self

    # ------------------------------------------- template method: transform
    def transform(self, series: pd.Series) -> pd.Series:
        self._check_is_fitted()
        s = self._to_numeric(series)
        mask = s.notna().to_numpy()

        out = np.full(s.shape[0], np.nan, dtype="float64")
        out[mask] = self._apply(s.to_numpy()[mask])

        if self.clip and self._clip_range_ is not None:
            lo, hi = self._clip_range_
            out[mask] = np.clip(out[mask], lo, hi)

        return self._build(out, s)

    # ----------------------------------- template method: inverse_transform
    def inverse_transform(self, series: pd.Series) -> pd.Series:
        self._check_is_fitted()
        s = self._to_numeric(series)
        mask = s.notna().to_numpy()

        out = np.full(s.shape[0], np.nan, dtype="float64")
        out[mask] = self._revert(s.to_numpy()[mask])
        return self._build(out, s)

    # ------------------------------------------------------------ niceties
    @property
    def params_(self) -> dict:
        """Learned statistics (everything ending in '_')."""
        self._check_is_fitted()
        return {
            k: v for k, v in vars(self).items()
            if k.endswith("_") and not k.startswith("_")
        }

    def __repr__(self) -> str:
        if not self._is_fitted:
            return f"{type(self).__name__}(not fitted)"
        body = ", ".join(
            f"{k.rstrip('_')}={v:.4g}" if isinstance(v, float) else f"{k.rstrip('_')}={v}"
            for k, v in self.params_.items()
        )
        return f"{type(self).__name__}({body})"
    
    
class MinMaxScaler(BaseScaler):
    """x' = (x - min) / (max - min) * (hi - lo) + lo   →  default range [0, 1]"""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0), **kw):
        super().__init__(**kw)
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError("feature_range must be (low, high) with low < high")
        self.feature_range = feature_range
        self.min_ = self.max_ = self.range_ = None

    def _learn(self, v):
        self.min_, self.max_ = float(v.min()), float(v.max())
        self.range_ = self._safe_denominator(self.max_ - self.min_, "max - min")

    def _apply(self, v):
        lo, hi = self.feature_range
        return (v - self.min_) / self.range_ * (hi - lo) + lo

    def _revert(self, v):
        lo, hi = self.feature_range
        return (v - lo) / (hi - lo) * self.range_ + self.min_


class StandardScaler(BaseScaler):
    """Z-score: x' = (x - mean) / std"""

    def __init__(self, ddof: int = 0, with_mean: bool = True,
                 with_std: bool = True, **kw):
        super().__init__(**kw)
        self.ddof, self.with_mean, self.with_std = ddof, with_mean, with_std
        self.mean_ = self.std_ = None

    def _learn(self, v):
        self.mean_ = float(v.mean()) if self.with_mean else 0.0
        self.std_ = (self._safe_denominator(float(v.std(ddof=self.ddof)), "std")
                     if self.with_std else 1.0)

    def _apply(self, v):
        return (v - self.mean_) / self.std_

    def _revert(self, v):
        return v * self.std_ + self.mean_


class RobustScaler(BaseScaler):
    """Outlier-resistant: x' = (x - median) / IQR"""

    def __init__(self, quantile_range: Tuple[float, float] = (25.0, 75.0), **kw):
        super().__init__(**kw)
        self.quantile_range = quantile_range
        self.center_ = self.iqr_ = None

    def _learn(self, v):
        q_lo, q_hi = np.percentile(v, self.quantile_range)
        self.center_ = float(np.median(v))
        self.iqr_ = self._safe_denominator(float(q_hi - q_lo), "IQR")

    def _apply(self, v):
        return (v - self.center_) / self.iqr_

    def _revert(self, v):
        return v * self.iqr_ + self.center_


class MaxAbsScaler(BaseScaler):
    """x' = x / max(|x|)  →  maps to [-1, 1] and preserves sparsity/zeros."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.max_abs_ = None

    def _learn(self, v):
        self.max_abs_ = self._safe_denominator(float(np.abs(v).max()), "max|x|")

    def _apply(self, v):
        return v / self.max_abs_

    def _revert(self, v):
        return v * self.max_abs_


class UnitNormScaler(BaseScaler):
    """Vector normalization: x' = x / ||x||_p  (whole Series treated as a vector)."""

    def __init__(self, norm: str = "l2", **kw):
        super().__init__(**kw)
        if norm not in {"l1", "l2", "max"}:
            raise ValueError("norm must be 'l1', 'l2' or 'max'")
        self.norm = norm
        self.norm_ = None

    def _learn(self, v):
        n = {"l1": np.abs(v).sum(),
             "l2": np.sqrt((v ** 2).sum()),
             "max": np.abs(v).max()}[self.norm]
        self.norm_ = self._safe_denominator(float(n), f"{self.norm} norm")

    def _apply(self, v):
        return v / self.norm_

    def _revert(self, v):
        return v * self.norm_
    
    
    
class ScalerFactory:
    _registry = {
        "minmax":   MinMaxScaler,
        "standard": StandardScaler,
        "zscore":   StandardScaler,
        "robust":   RobustScaler,
        "maxabs":   MaxAbsScaler,
        "unit":     UnitNormScaler,
    }

    @classmethod
    def create(cls, method: str, **kwargs) -> BaseScaler:
        try:
            return cls._registry[method.lower()](**kwargs)
        except KeyError:
            raise ValueError(
                f"Unknown method {method!r}. Available: {sorted(cls._registry)}"
            ) from None

    @classmethod
    def register(cls, name: str, klass: type) -> None:
        """Add your own scaler without touching this file."""
        if not issubclass(klass, BaseScaler):
            raise TypeError("klass must subclass BaseScaler")
        cls._registry[name.lower()] = klass
        
        

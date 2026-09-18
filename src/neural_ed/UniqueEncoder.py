from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


class NotFittedError(Exception):
    """Raised when transform() is called before fit()."""


class DuplicateValueError(Exception):
    """Raised when a series expected to contain only unique values does not."""


class BaseEncoder(ABC):
    """Minimal scikit-learn-like contract for every encoder we write."""

    def __init__(self) -> None:
        self._is_fitted: bool = False

    # ---------- abstract API ----------
    @abstractmethod
    def fit(self, series: pd.Series) -> "BaseEncoder":
        ...

    @abstractmethod
    def transform(self, series: pd.Series) -> pd.Series:
        ...

    @abstractmethod
    def inverse_transform(self, series: pd.Series) -> pd.Series:
        ...

    # ---------- shared helpers ----------
    def fit_transform(self, series: pd.Series) -> pd.Series:
        return self.fit(series).transform(series)

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError(
                f"{type(self).__name__} is not fitted yet. "
                "Call fit() or fit_transform() first."
            )

    @staticmethod
    def _validate_series(series) -> pd.Series:
        if not isinstance(series, pd.Series):
            raise TypeError(f"Expected pd.Series, got {type(series).__name__}")
        return series


class UniqueEncoder(BaseEncoder):
    """
    Encodes a Series whose non-null values are expected to be *unique*
    (e.g. an ID column, a primary key, a slug) into integer codes and
    back again.

    Unlike OrdinalEncoder, this is not meant for repeated categorical
    labels — it is a 1:1 mapping between each distinct value and a code.
    By default it enforces uniqueness at fit time and raises if it finds
    duplicates, since a repeated value would indicate the column isn't
    actually "unique-valued" as assumed.

    Codes are assigned in order of first appearance (no sorting), since
    there is no notion of category ordering for unique identifiers.
    """

    def __init__(
        self,
        start: int = 0,
        handle_unknown: str = "error",
        unknown_value: int | float | None = None,
        encode_missing_as: int | float = np.nan,
        enforce_unique: bool = True,
        dtype: str | type = np.int64,
    ) -> None:
        super().__init__()

        if handle_unknown not in {"error", "use_encoded_value"}:
            raise ValueError("handle_unknown must be 'error' or 'use_encoded_value'")
        if handle_unknown == "use_encoded_value" and unknown_value is None:
            raise ValueError(
                "unknown_value must be set when handle_unknown='use_encoded_value'"
            )

        self.start = start
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encode_missing_as = encode_missing_as
        self.enforce_unique = enforce_unique
        self.dtype = dtype

        # learned attributes (trailing underscore = set during fit)
        self.categories_: np.ndarray | None = None
        self.mapping_: dict | None = None
        self.inverse_mapping_: dict | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, series: pd.Series) -> "UniqueEncoder":
        series = self._validate_series(series)

        non_null = series.dropna()

        if self.enforce_unique:
            dupes = non_null[non_null.duplicated(keep=False)]
            if not dupes.empty:
                bad = sorted(map(str, dupes.unique()))
                raise DuplicateValueError(
                    f"UniqueEncoder requires unique values but found duplicates: {bad}"
                )

        # order of first appearance, not sorted — there's no natural
        # ordering among unique identifiers
        values = non_null.tolist()
        seen = dict.fromkeys(values)  # dedupes while preserving order
        categories = list(seen.keys())

        self.categories_ = np.array(categories, dtype=object)
        self.mapping_ = {c: i + self.start for i, c in enumerate(categories)}
        self.inverse_mapping_ = {i: c for c, i in self.mapping_.items()}
        self._is_fitted = True
        return self

    # ------------------------------------------------------------ transform
    def transform(self, series: pd.Series) -> pd.Series:
        self._check_is_fitted()
        series = self._validate_series(series)

        known = series.isin(self.mapping_.keys())
        unknown_mask = ~known & series.notna()

        if unknown_mask.any():
            if self.handle_unknown == "error":
                bad = sorted(map(str, series[unknown_mask].unique()))
                raise ValueError(f"Found unknown values during transform: {bad}")

        codes = series.map(self.mapping_)                     # unseen -> NaN
        codes[unknown_mask] = self.unknown_value              # unseen -> sentinel
        codes[series.isna()] = self.encode_missing_as         # NaN     -> sentinel

        return pd.Series(
            codes.values, index=series.index,
            name=series.name, dtype=self.dtype,
        )

    # ---------------------------------------------------- inverse_transform
    def inverse_transform(self, series: pd.Series) -> pd.Series:
        self._check_is_fitted()
        series = self._validate_series(series)

        decoded = series.map(
            lambda v: self.inverse_mapping_.get(int(v)) if pd.notna(v) else np.nan
        )
        return pd.Series(
            decoded.values, index=series.index,
            name=series.name, dtype=object,
        )

    # ------------------------------------------------------------- niceties
    def get_params(self) -> dict:
        return {
            "start": self.start,
            "handle_unknown": self.handle_unknown,
            "unknown_value": self.unknown_value,
            "encode_missing_as": self.encode_missing_as,
            "enforce_unique": self.enforce_unique,
            "dtype": self.dtype,
        }

    def __len__(self) -> int:
        self._check_is_fitted()
        return len(self.categories_)

    def __repr__(self) -> str:
        state = (
            f"categories_={list(self.categories_)}"
            if self._is_fitted else "not fitted"
        )
        return f"UniqueEncoder({state})"
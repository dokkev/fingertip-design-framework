"""Tests for shared scalar validation helpers."""

from __future__ import annotations

import math

import pytest

from util import require_finite, require_nonnegative, require_positive


def test_require_finite_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="length must be finite"):
        require_finite("length", math.nan)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan])
def test_require_positive_rejects_nonpositive_and_nan_values(value: float) -> None:
    with pytest.raises(ValueError, match="length must be greater than zero"):
        require_positive("length", value)


@pytest.mark.parametrize("value", [-1.0, math.nan])
def test_require_nonnegative_rejects_negative_and_nan_values(value: float) -> None:
    with pytest.raises(ValueError, match="weight must be nonnegative"):
        require_nonnegative("weight", value)


def test_range_helpers_support_custom_error_types() -> None:
    class CustomError(ValueError):
        pass

    with pytest.raises(CustomError):
        require_positive("length", 0.0, error_type=CustomError)
    with pytest.raises(CustomError):
        require_nonnegative("weight", -1.0, error_type=CustomError)

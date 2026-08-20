"""Small, explicit scalar validation helpers."""

from __future__ import annotations

from math import isfinite
from typing import TypeVar


_ErrorT = TypeVar("_ErrorT", bound=Exception)


def require_finite(
    name: str,
    value: float,
    *,
    error_type: type[_ErrorT] = ValueError,
) -> None:
    """Raise ``error_type`` when ``value`` is not finite."""
    if not isfinite(value):
        raise error_type(f"{name} must be finite")


def require_positive(
    name: str,
    value: float,
    *,
    error_type: type[_ErrorT] = ValueError,
) -> None:
    """Raise ``error_type`` when ``value`` is not greater than zero."""
    require_finite(name, value, error_type=error_type)
    if value <= 0.0:
        raise error_type(f"{name} must be greater than zero")


def require_nonnegative(
    name: str,
    value: float,
    *,
    error_type: type[_ErrorT] = ValueError,
) -> None:
    """Raise ``error_type`` when ``value`` is less than zero or non-finite."""
    require_finite(name, value, error_type=error_type)
    if value < 0.0:
        raise error_type(f"{name} must be greater than or equal to zero")


__all__ = ["require_finite", "require_nonnegative", "require_positive"]

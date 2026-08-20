"""Shared scalar validation helpers."""

from __future__ import annotations

from math import isfinite


def require_finite(
    name: str,
    value: float,
    *,
    error_type: type[ValueError] = ValueError,
) -> None:
    """Raise ``error_type`` when a scalar value is not finite."""
    if not isfinite(value):
        raise error_type(f"{name} must be finite")


def require_positive(
    name: str,
    value: float,
    *,
    error_type: type[ValueError] = ValueError,
) -> None:
    """Raise ``error_type`` when a scalar value is not greater than zero."""
    if not value > 0.0:
        raise error_type(f"{name} must be greater than zero")


def require_nonnegative(
    name: str,
    value: float,
    *,
    error_type: type[ValueError] = ValueError,
) -> None:
    """Raise ``error_type`` when a scalar value is below zero."""
    if not value >= 0.0:
        raise error_type(f"{name} must be nonnegative")


__all__ = ["require_finite", "require_nonnegative", "require_positive"]

"""Shared validation primitives for model value objects."""

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


__all__ = ["require_finite"]

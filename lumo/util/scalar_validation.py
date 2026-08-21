"""Small dependency-free validation helpers for scalar values."""

from __future__ import annotations

from math import isfinite
from numbers import Real


def _is_finite_real(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and isfinite(value)
    )


def require_finite(
    name: str,
    value: float,
    *,
    error_type: type[ValueError] = ValueError,
) -> None:
    """Require a finite real scalar."""

    if not _is_finite_real(value):
        raise error_type(f"{name} must be finite, got {value!r}")


def require_positive(
    name: str,
    value: float,
    *,
    error_type: type[ValueError] = ValueError,
) -> None:
    """Require a finite positive scalar."""

    if not _is_finite_real(value) or value <= 0.0:
        raise error_type(
            f"{name} must be finite and positive, got {value!r}"
        )


def require_nonnegative(
    name: str,
    value: float,
    *,
    error_type: type[ValueError] = ValueError,
) -> None:
    """Require a finite non-negative scalar."""

    if not _is_finite_real(value) or value < 0.0:
        raise error_type(
            f"{name} must be finite and non-negative, got {value!r}"
        )


__all__ = [
    "require_finite",
    "require_nonnegative",
    "require_positive",
]

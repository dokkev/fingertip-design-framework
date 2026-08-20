from __future__ import annotations

import pytest

from util.validation import require_finite, require_nonnegative, require_positive


def test_require_finite_accepts_finite_values() -> None:
    require_finite("value", 1.0)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_require_finite_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="value must be finite"):
        require_finite("value", value)


def test_require_finite_preserves_domain_error_type() -> None:
    class DomainError(ValueError):
        pass

    with pytest.raises(DomainError, match="length must be finite"):
        require_finite("length", float("nan"), error_type=DomainError)


@pytest.mark.parametrize("value", (0.0, -1.0))
def test_require_positive_rejects_non_positive_values(value: float) -> None:
    with pytest.raises(ValueError, match="value must be greater than zero"):
        require_positive("value", value)


@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_require_positive_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="value must be finite"):
        require_positive("value", value)


def test_require_nonnegative_accepts_zero() -> None:
    require_nonnegative("value", 0.0)


def test_require_nonnegative_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        require_nonnegative("value", -1.0)


@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_require_nonnegative_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="value must be finite"):
        require_nonnegative("value", value)

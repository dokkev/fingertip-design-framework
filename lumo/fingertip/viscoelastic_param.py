"""Viscoelastic material parameters for the LUMO fingertip."""

from __future__ import annotations

from dataclasses import dataclass
from lumo.util.scalar_validation import require_nonnegative, require_positive
from .geometric_param import InvalidFingertipParameters


@dataclass(frozen=True)
class ViscoelasticParameters:
    """Mechanical material parameters for the compliant silicone pad."""

    density_kg_m3: float = 1070.0
    k_mu_pa: float = 1.06e5
    k_lambda_pa: float = 1.0494e7
    damping: float = 10.0

    def __post_init__(self) -> None:
        error_type = InvalidFingertipParameters

        require_positive(
            "density_kg_m3",
            self.density_kg_m3,
            error_type=error_type,
        )
        require_positive(
            "k_mu_pa",
            self.k_mu_pa,
            error_type=error_type,
        )
        require_positive(
            "k_lambda_pa",
            self.k_lambda_pa,
            error_type=error_type,
        )
        require_nonnegative(
            "damping",
            self.damping,
            error_type=error_type,
        )


__all__ = ["ViscoelasticParameters"]

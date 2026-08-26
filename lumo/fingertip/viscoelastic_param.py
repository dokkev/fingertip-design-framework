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
    
    # dragon_skin_10nv_datasheet
    #     density = 1070 kg/m^3
    #     k_mu    = 106376 Pa
    #     nu      = existing silicone nu
    #     k_lambda = derived from k_mu and existing nu
    #     k_damp  = existing silicone k_damp

    # solaris_datasheet
    #     density = 990 kg/m^3
    #     k_mu    = 98497 Pa
    #     nu      = existing silicone nu
    #     k_lambda = derived from k_mu and existing nu
    #     k_damp  = existing silicone k_damp
    
    

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


SILICONE_VISCOELASTIC = ViscoelasticParameters()

VISCOELASTIC_PRESETS = {
    "silicone": SILICONE_VISCOELASTIC,
}


__all__ = [
    "SILICONE_VISCOELASTIC",
    "VISCOELASTIC_PRESETS",
    "ViscoelasticParameters",
]

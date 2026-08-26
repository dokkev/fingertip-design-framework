"""Mechanical material parameters for the LUMO silicone fingertip."""

from __future__ import annotations

from dataclasses import dataclass

from lumo.util.scalar_validation import require_nonnegative, require_positive


@dataclass(frozen=True)
class SiliconeMechanics:
    """Damped Neo-Hookean inputs consumed directly by Newton."""

    density_kg_m3: float = 1070.0
    shear_modulus_pa: float = 1.06e5
    lame_lambda_pa: float = 1.0494e7
    damping_pa_s: float = 10.0

    def __post_init__(self) -> None:
        require_positive("density_kg_m3", self.density_kg_m3)
        require_positive("shear_modulus_pa", self.shear_modulus_pa)
        require_positive("lame_lambda_pa", self.lame_lambda_pa)
        require_nonnegative("damping_pa_s", self.damping_pa_s)


SILICONE_MECHANICS = SiliconeMechanics()

MECHANICS_PRESETS = {
    "silicone": SILICONE_MECHANICS,
}


__all__ = [
    "MECHANICS_PRESETS",
    "SILICONE_MECHANICS",
    "SiliconeMechanics",
]

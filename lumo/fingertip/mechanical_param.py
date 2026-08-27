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

# Smooth-On reports 0.99 g/cc cured density and 25 psi stress at 100%
# elongation for Solaris.  For an incompressible Neo-Hookean solid at stretch
# 2, P = 1.75 * mu, giving mu ~= 98.5 kPa.  Lambda retains the existing
# numerical Poisson ratio of approximately 0.495.  The damping remains an
# uncalibrated numerical input because the datasheet does not report it.
SOLARIS_MECHANICS = SiliconeMechanics(
    density_kg_m3=990.0,
    shear_modulus_pa=9.85e4,
    lame_lambda_pa=9.75e6,
    damping_pa_s=10.0,
)

MECHANICS_PRESETS = {
    "silicone": SILICONE_MECHANICS,
    "solaris": SOLARIS_MECHANICS,
}


__all__ = [
    "MECHANICS_PRESETS",
    "SILICONE_MECHANICS",
    "SOLARIS_MECHANICS",
    "SiliconeMechanics",
]

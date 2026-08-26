"""Optical material and LED source parameters for the LUMO fingertip."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from lumo.util.scalar_validation import require_nonnegative, require_positive

from .geometric_param import InvalidFingertipParameters


@dataclass(frozen=True)
class SiliconeOptics:
    """Effective monochromatic optical properties of one silicone."""

    name: str
    refractive_index: float
    extinction_coefficient_m_inv: float

    def __post_init__(self) -> None:
        error_type = InvalidFingertipParameters

        if not isinstance(self.name, str) or not self.name.strip():
            raise error_type("name must be a nonempty string")

        require_positive(
            "refractive_index",
            self.refractive_index,
            error_type=error_type,
        )
        require_nonnegative(
            "extinction_coefficient_m_inv",
            self.extinction_coefficient_m_inv,
            error_type=error_type,
        )


# Optical-property presets are literature/manufacturer priors, not calibrated
# measurements of the specific fingertip specimens.
#
# Reported PDMS propagation loss A [dB/cm] is converted to the Beer-Lambert
# effective extinction coefficient used by LUMO as:
#
#   mu [m^-1] = A [dB/cm] * 100 * ln(10) / 10
#              = A * 23.02585093
#
# "Extinction" is used rather than pure absorption because reported waveguide
# losses can include bulk absorption, scattering, and fabrication-related loss.

# Solaris
# -------
# Smooth-On describes Solaris as clear/ultra-transparent and reports n = 1.41
# at 20 C (ASTM D-1218).
# Source: Smooth-On, Solaris Technical Bulletin.
#
# Because no product-specific spectral extinction coefficient is provided,
# LOW/NOMINAL/HIGH use the lower range of reported visible-light PDMS losses:
#
#   LOW:     0.36 dB/cm @ 635 nm ->  8.2893 m^-1
#            Ersen & Sahin, Journal of Biomedical Optics, 2017.
#            DOI: 10.1117/1.JBO.22.5.055005
#
#   NOMINAL: 0.63 dB/cm @ 441.6 nm -> 14.5063 m^-1
#            Guo et al., Polymers, 2019.
#            DOI: 10.3390/polym11091433
#
#   HIGH:    1.8 dB/cm @ 532 nm -> 41.4465 m^-1
#            Bliss et al., Lab on a Chip, 2007.
#            DOI: 10.1039/B708485D

SOLARIS_OPTICS_LOW = SiliconeOptics(
    name="Solaris",
    refractive_index=1.41,
    extinction_coefficient_m_inv=8.289306334778566,
)
SOLARIS_OPTICS_NOMINAL = SiliconeOptics(
    name="Solaris",
    refractive_index=1.41,
    extinction_coefficient_m_inv=14.50628608586249,
)
SOLARIS_OPTICS_HIGH = SiliconeOptics(
    name="Solaris",
    refractive_index=1.41,
    extinction_coefficient_m_inv=41.44653167389283,
)


# Dragon Skin 10 NV
# -----------------
# Smooth-On describes Dragon Skin 10 NV as "water white translucent", but does
# not provide a product-specific refractive index or spectral attenuation.
#
# n = 1.4348 is therefore a transparent-PDMS literature proxy near the LED
# wavelength (reported for Sylgard 184 at 532 nm), NOT a Dragon Skin-specific
# measurement.
#
# Since Dragon Skin 10 NV is translucent, its LOW/NOMINAL/HIGH extinction
# priors use the higher range of reported visible-light PDMS propagation loss:
#
#   LOW:     1.8 dB/cm @ 532 nm ->  41.4465 m^-1
#            Bliss et al., Lab on a Chip, 2007.
#            DOI: 10.1039/B708485D
#
#   NOMINAL: 3.1 dB/cm @ 532 nm -> 71.3801 m^-1
#            Azmayesh-Fard et al., J. Micromech. Microeng., 2010.
#            DOI: 10.1088/0960-1317/20/8/087002
#
#   HIGH:    4.8 dB/cm @ 473 nm -> 110.5241 m^-1
#            Rudmann et al., Advanced Healthcare Materials, 2024.
#            DOI: 10.1002/adhm.202304513
#
# These coefficients should be interpreted as effective optical-loss priors,
# not measured Dragon Skin absorption coefficients.
#
DRAGON_SKIN_10_NV_OPTICS_LOW = SiliconeOptics(
    name="Dragon Skin 10 NV",
    refractive_index=1.4348,
    extinction_coefficient_m_inv=41.44653167389283,
)
DRAGON_SKIN_10_NV_OPTICS_NOMINAL = SiliconeOptics(
    name="Dragon Skin 10 NV",
    refractive_index=1.4348,
    extinction_coefficient_m_inv=71.38013788281543,
)
DRAGON_SKIN_10_NV_OPTICS_HIGH = SiliconeOptics(
    name="Dragon Skin 10 NV",
    refractive_index=1.4348,
    extinction_coefficient_m_inv=110.52408446371422,
)

OPTICAL_PRESETS = {
    "solaris_low": SOLARIS_OPTICS_LOW,
    "solaris_nominal": SOLARIS_OPTICS_NOMINAL,
    "solaris_high": SOLARIS_OPTICS_HIGH,
    "dragon_skin_10_nv_low": DRAGON_SKIN_10_NV_OPTICS_LOW,
    "dragon_skin_10_nv_nominal": DRAGON_SKIN_10_NV_OPTICS_NOMINAL,
    "dragon_skin_10_nv_high": DRAGON_SKIN_10_NV_OPTICS_HIGH,
}


@dataclass(frozen=True)
class LEDParameters:
    """Current Green Sequin design envelope and optical source parameters."""

    ADAFRUIT_PRODUCT_ID: ClassVar[int] = 1756
    LED_PART_NUMBER: ClassVar[str] = "LuckyLight S150PGC-G5-1B"
    PACKAGE: ClassVar[str] = "1206 Pure Green InGaN"

    # The 2-D fingertip model uses the 4 mm by 2 mm board cross-section. The
    # board's omitted out-of-plane dimension is 9 mm.
    width_mm: float = 4.0
    height_mm: float = 2.0
    # LuckyLight package drawing: water-clear resin emitting window.  These
    # dimensions define the modeled finite aperture, not calibrated radiometry.
    emitting_window_x_mm: float = 1.8
    emitting_window_y_mm: float = 1.6
    # Modeled source power before absolute optical calibration.
    normalized_power: float = 1.0
    dominant_wavelength_nm: float = 525.0
    peak_wavelength_nm: float = 520.0
    spectral_half_width_nm: float = 35.0
    # The datasheet specifies a 120 degree full half-intensity viewing angle.
    viewing_half_angle_deg: float = 60.0

    def __post_init__(self) -> None:
        error_type = InvalidFingertipParameters

        require_positive(
            "width_mm",
            self.width_mm,
            error_type=error_type,
        )
        require_positive(
            "height_mm",
            self.height_mm,
            error_type=error_type,
        )
        require_positive(
            "emitting_window_x_mm",
            self.emitting_window_x_mm,
            error_type=error_type,
        )
        require_positive(
            "emitting_window_y_mm",
            self.emitting_window_y_mm,
            error_type=error_type,
        )
        require_nonnegative(
            "normalized_power",
            self.normalized_power,
            error_type=error_type,
        )
        for name in (
            "dominant_wavelength_nm",
            "peak_wavelength_nm",
            "spectral_half_width_nm",
        ):
            require_positive(
                name,
                getattr(self, name),
                error_type=error_type,
            )

        if not 0.0 < self.viewing_half_angle_deg < 90.0:
            raise error_type("viewing_half_angle_deg must be between 0 and 90 degrees")


__all__ = [
    "DRAGON_SKIN_10_NV_OPTICS_HIGH",
    "DRAGON_SKIN_10_NV_OPTICS_LOW",
    "DRAGON_SKIN_10_NV_OPTICS_NOMINAL",
    "LEDParameters",
    "OPTICAL_PRESETS",
    "SOLARIS_OPTICS_HIGH",
    "SOLARIS_OPTICS_LOW",
    "SOLARIS_OPTICS_NOMINAL",
    "SiliconeOptics",
]

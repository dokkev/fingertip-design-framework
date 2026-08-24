"""Pure validation and correction diagnostics for the design-space GUI."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from numbers import Real
from typing import Literal, Mapping

from lumo.finger import (
    Fingertip,
    FingertipParameters,
    KinematicParameters,
    LED,
    OpticalParameters,
    PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM,
    ellipse_depth_at_cutout_mm,
    silicone_thickness_measures,
)
from lumo.optimization.design_space import OPTIMIZABLE_PARAMETER_NAMES


DiagnosticSeverity = Literal["INFO", "WARN", "ERROR"]


@dataclass(frozen=True)
class Diagnostic:
    """One current-state console message."""

    severity: DiagnosticSeverity
    source: str
    message: str

    @property
    def formatted(self) -> str:
        return f"[{self.severity}][{self.source}] {self.message}"


_PARAMETER_DEFAULTS = {
    field.name: getattr(KinematicParameters(), field.name)
    for field in fields(KinematicParameters)
}
_LED_DEFAULTS = {
    field.name: getattr(LED(), field.name) for field in fields(LED)
}
_OPTICAL_DEFAULTS = {
    field.name: getattr(OpticalParameters(), field.name)
    for field in fields(OpticalParameters)
}
_PRIMARY_DIMENSIONS = (
    "flat_pad_width",
    "flat_pad_height",
    "semielliptical_pad_height",
    "link_thickness",
    "bond_extension_width",
    "bond_extension_height",
    "stem_width",
    "stem_height",
)
_VOID_DIMENSIONS = ("void_width", "void_height")


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    resolved = float(value)
    return resolved if isfinite(resolved) else None


def _display(value: object, unit: str = "") -> str:
    number = _number(value)
    if number is None:
        return f"{value!r}{unit}"
    return f"{number:g}{unit}"


def _merged(defaults: Mapping[str, object], values: Mapping[str, object]) -> dict[str, object]:
    merged = dict(defaults)
    merged.update(values)
    return merged


def _message(
    severity: DiagnosticSeverity,
    source: str,
    message: str,
) -> Diagnostic:
    return Diagnostic(severity=severity, source=source, message=message)


def _geometry_corrections(values: Mapping[str, object]) -> list[Diagnostic]:
    """Compute known correction ranges without replacing model validation."""
    result: list[Diagnostic] = []
    numeric = {name: _number(values.get(name)) for name in _PARAMETER_DEFAULTS}

    for name in _PRIMARY_DIMENSIONS:
        value = numeric[name]
        if value is not None and value <= 0.0:
            result.append(
                _message(
                    "ERROR",
                    "GEOMETRY",
                    f"{name} must be > 0 mm; current value is {value:g} mm.",
                )
            )
    for name in _VOID_DIMENSIONS:
        value = numeric[name]
        if value is not None and value < 0.0:
            result.append(
                _message(
                    "ERROR",
                    "GEOMETRY",
                    f"{name} must be >= 0 mm; current value is {value:g} mm.",
                )
            )

    required = (
        numeric["bond_extension_width"],
        numeric["stem_width"],
        numeric["void_width"],
        numeric["flat_pad_width"],
    )
    if all(value is not None for value in required):
        bond_width, stem_width, void_width, flat_width = required
        width_minimum = 2.0 * bond_width + stem_width + 2.0 * void_width
        if width_minimum >= flat_width:
            result.append(
                _message(
                    "ERROR",
                    "GEOMETRY",
                    "Horizontal width fit is violated. Current values: "
                    f"flat_pad_width={flat_width:g} mm, "
                    f"bond_extension_width={bond_width:g} mm, "
                    f"stem_width={stem_width:g} mm, "
                    f"void_width={void_width:g} mm.\n"
                    "Constraint: flat_pad_width > "
                    "2*bond_extension_width + stem_width + 2*void_width.\n"
                    f"Required: flat_pad_width > {width_minimum:g} mm; "
                    f"stem_width < {flat_width - 2.0 * bond_width - 2.0 * void_width:g} mm.",
                )
            )
            if flat_width - stem_width - 2.0 * void_width > 0.0:
                result.append(
                    _message(
                        "INFO",
                        "GEOMETRY",
                        "For the current flat_pad_width and clearances, "
                        "bond_extension_width must be < "
                        f"{(flat_width - stem_width - 2.0 * void_width) / 2.0:g} mm.",
                    )
                )
            else:
                result.append(
                    _message(
                        "ERROR",
                        "GEOMETRY",
                        "The current fixed dimensions leave no feasible positive "
                        "stem width.",
                    )
                )

    bond_height = numeric["bond_extension_height"]
    link_thickness = numeric["link_thickness"]
    if bond_height is not None and link_thickness is not None:
        if bond_height >= link_thickness:
            result.append(
                _message(
                    "ERROR",
                    "GEOMETRY",
                    f"bond_extension_height must be < {link_thickness:g} mm "
                    f"(current {bond_height:g} mm), or link_thickness must be "
                    f"> {bond_height:g} mm.",
                )
            )

    flat_width = numeric["flat_pad_width"]
    stem_width = numeric["stem_width"]
    void_width = numeric["void_width"]
    tolerance = numeric["geometry_length_tolerance_mm"]
    if (
        flat_width is not None
        and stem_width is not None
        and void_width is not None
        and tolerance is not None
        and flat_width > 0.0
    ):
        cutout_half_width = stem_width / 2.0 + void_width
        available_half_width = flat_width / 2.0 - tolerance
        if cutout_half_width >= available_half_width:
            result.append(
                _message(
                    "ERROR",
                    "GEOMETRY",
                    "The cutout reaches the external half-width. Current values: "
                    f"cutout_half_width={cutout_half_width:g} mm, "
                    "flat_pad_width/2 - geometry_length_tolerance_mm="
                    f"{available_half_width:g} mm.\n"
                    "Constraint: cutout_half_width < "
                    "flat_pad_width/2 - geometry_length_tolerance_mm.\n"
                    f"Required: flat_pad_width > {2.0 * (cutout_half_width + tolerance):g} mm "
                    f"or stem_width < {flat_width - 2.0 * void_width - 2.0 * tolerance:g} mm.",
                )
            )

    ellipse_inputs = (
        flat_width,
        numeric["flat_pad_height"],
        numeric["semielliptical_pad_height"],
        stem_width,
        void_width,
        numeric["stem_height"],
        numeric["void_height"],
        tolerance,
    )
    if all(value is not None for value in ellipse_inputs):
        (
            flat_width,
            flat_height,
            ellipse_height,
            stem_width,
            void_width,
            stem_height,
            void_height,
            tolerance,
        ) = ellipse_inputs
        if flat_width > 0.0:
            half_width = flat_width / 2.0
            cutout_half_width = stem_width / 2.0 + void_width
            normalized_x = cutout_half_width / half_width
            if 0.0 <= normalized_x < 1.0:
                available_depth = ellipse_depth_at_cutout_mm(
                    flat_pad_width=flat_width,
                    semielliptical_pad_height=ellipse_height,
                    stem_width=stem_width,
                    void_width=void_width,
                )
                penetration = max(0.0, stem_height + void_height - flat_height)
                if penetration > 0.0 and penetration >= available_depth - tolerance:
                    max_stem_height = (
                        flat_height
                        - void_height
                        + available_depth
                        - tolerance
                    )
                    required_ellipse_height = (
                        (penetration + tolerance)
                        / (available_depth / ellipse_height)
                        if available_depth > 0.0 and ellipse_height > 0.0
                        else None
                    )
                    result.append(
                        _message(
                            "ERROR",
                            "GEOMETRY",
                            "Cutout bottom exits the semielliptical envelope.\n"
                            f"Current penetration={penetration:g} mm; "
                            f"available ellipse depth={available_depth:g} mm; "
                            f"geometry_length_tolerance_mm={tolerance:g} mm.\n"
                            "Constraint: penetration < available ellipse depth "
                            "- geometry_length_tolerance_mm.\n"
                            f"Possible fixes: stem_height < {max_stem_height:g} mm; "
                            + (
                                f"semielliptical_pad_height > {required_ellipse_height:g} mm; "
                                if required_ellipse_height is not None
                                else "increase semielliptical_pad_height; "
                            )
                            + "or increase flat_pad_height.",
                        )
                    )
                    required_flat_height = (
                        stem_height + void_height - available_depth + tolerance
                    )
                    result.append(
                        _message(
                            "INFO",
                            "GEOMETRY",
                            f"The current cutout requires flat_pad_height > "
                            f"{required_flat_height:g} mm for this ellipse depth.",
                        )
                    )
            elif cutout_half_width > 0.0:
                result.append(
                    _message(
                        "ERROR",
                        "GEOMETRY",
                        f"The cutout half-width {cutout_half_width:g} mm is not "
                        f"strictly inside flat_pad_width/2={half_width:g} mm; "
                        "reduce stem/clearance width or increase overall width.",
                    )
                )
    return result


def diagnose_geometry(
    values: Mapping[str, object],
    *,
    led: LED | None = None,
) -> tuple[Diagnostic, ...]:
    """Validate actual fingertip construction and add known geometry guidance."""
    payload = _merged(_PARAMETER_DEFAULTS, values)
    result = _geometry_corrections(payload)
    try:
        parameters = FingertipParameters(**payload)
    except Exception as exc:
        result.append(
            _message(
                "ERROR",
                "GEOMETRY",
                f"{type(exc).__name__}: {exc}\n"
                "No automatic correction was applied; inspect the constraints above.",
            )
        )
        return tuple(result)

    thickness = silicone_thickness_measures(parameters)
    result.append(
        _message(
            "INFO",
            "GEOMETRY",
            "Production silicone wall thickness: "
            f"global_d_min={thickness.minimum_silicone_thickness_mm:g} mm, "
            f"boundary_pair={thickness.shortest_boundary_pair}.",
        )
    )
    minimum_thickness_mm = PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM
    if thickness.minimum_silicone_thickness_mm < minimum_thickness_mm:
        result.append(
            _message(
                "ERROR",
                "DESIGN SPACE",
                "The production "
                f"{minimum_thickness_mm:g} mm silicone wall-thickness rule "
                "is violated: "
                f"global_d_min={thickness.minimum_silicone_thickness_mm:g} mm, "
                f"boundary_pair={thickness.shortest_boundary_pair}. "
                "Increase the surrounding pad dimensions or reduce the cutout; "
                "no automatic repair was applied.",
            )
        )

    selected_led = led or LED()
    try:
        Fingertip(parameters, led=selected_led)
    except Exception as exc:
        result.append(
            _message(
                "ERROR",
                "GEOMETRY",
                f"{type(exc).__name__}: {exc}\n"
                "The physical constructor rejected this fingertip; no automatic "
                "repair was applied.",
            )
        )
        if (
            selected_led.width_mm
            > parameters.stem_width + parameters.geometry_length_tolerance_mm
        ):
            result.append(
                _message(
                    "ERROR",
                    "LED FIT",
                    f"LED width={selected_led.width_mm:g} mm exceeds stem width="
                    f"{parameters.stem_width:g} mm. Required relation: "
                    "LED width <= stem_width + geometry_length_tolerance_mm="
                    f"{parameters.stem_width + parameters.geometry_length_tolerance_mm:g} mm. "
                    "Increase stem_width or reduce LED width.",
                )
            )
        if (
            selected_led.height_mm
            > parameters.stem_height + parameters.geometry_length_tolerance_mm
        ):
            result.append(
                _message(
                    "ERROR",
                    "LED FIT",
                    f"LED height={selected_led.height_mm:g} mm exceeds stem height="
                    f"{parameters.stem_height:g} mm. Required relation: "
                    "LED height <= stem_height + geometry_length_tolerance_mm="
                    f"{parameters.stem_height + parameters.geometry_length_tolerance_mm:g} mm. "
                    "Increase stem_height or reduce LED height.",
                )
            )
    return tuple(result)


def diagnose_led(values: Mapping[str, object]) -> tuple[Diagnostic, ...]:
    """Report model-supported LED value intervals."""
    payload = _merged(_LED_DEFAULTS, values)
    result: list[Diagnostic] = []
    for name in ("width_mm", "height_mm"):
        value = _number(payload[name])
        if value is None or value <= 0.0:
            result.append(
                _message(
                    "ERROR",
                    "LED",
                    f"{name} must be > 0 mm; current value is {_display(payload[name], ' mm')}.",
                )
            )
    power = _number(payload["relative_radiant_power"])
    if power is None or power < 0.0:
        result.append(
            _message(
                "ERROR",
                "LED",
                "relative_radiant_power must be >= 0; current value is "
                f"{_display(payload['relative_radiant_power'])}.",
            )
        )
    angle = _number(payload["emission_half_angle_deg"])
    if angle is None or not 0.0 < angle < 90.0:
        result.append(
            _message(
                "ERROR",
                "LED",
                "emission_half_angle_deg must satisfy 0 < angle < 90; "
                f"current value is {_display(payload['emission_half_angle_deg'])} deg.",
            )
        )
    try:
        LED(**payload)
    except Exception as exc:
        result.append(_message("ERROR", "LED", f"{type(exc).__name__}: {exc}"))
    return tuple(result)


def diagnose_optical(values: Mapping[str, object]) -> tuple[Diagnostic, ...]:
    """Report model-supported optical material intervals without tracing rays."""
    payload = _merged(_OPTICAL_DEFAULTS, values)
    result: list[Diagnostic] = []
    for name in ("refractive_index_air", "refractive_index_silicone"):
        value = _number(payload[name])
        if value is None or value <= 0.0:
            result.append(
                _message(
                    "ERROR",
                    "OPTICAL",
                    f"{name} must be > 0; current value is {_display(payload[name])}.",
                )
            )
    absorption = _number(payload["absorption_per_mm"])
    if absorption is None or absorption < 0.0:
        result.append(
            _message(
                "ERROR",
                "OPTICAL",
                "absorption_per_mm must be >= 0; current value is "
                f"{_display(payload['absorption_per_mm'])}.",
            )
        )
    try:
        OpticalParameters(**payload)
    except Exception as exc:
        result.append(_message("ERROR", "OPTICAL", f"{type(exc).__name__}: {exc}"))
    return tuple(result)


def diagnose_physical_state(
    geometry: Mapping[str, object],
    led_values: Mapping[str, object],
) -> tuple[Diagnostic, ...]:
    """Validate geometry and LED fit for one state."""
    led_diagnostics = diagnose_led(led_values)
    selected_led: LED | None = None
    if not led_diagnostics:
        selected_led = LED(**_merged(_LED_DEFAULTS, led_values))
    return (
        *diagnose_geometry(geometry, led=selected_led),
        *led_diagnostics,
    )


def diagnose_state(
    geometry: Mapping[str, object],
    led_values: Mapping[str, object],
    optical: Mapping[str, object],
) -> tuple[Diagnostic, ...]:
    """Validate all editable state; optical checks remain separate from shape."""
    return (
        *diagnose_physical_state(geometry, led_values),
        *diagnose_optical(optical),
    )


def diagnose_design_space(
    nominal_parameters: Mapping[str, object],
    variables: Mapping[str, Mapping[str, object]],
) -> tuple[Diagnostic, ...]:
    """Report incomplete or out-of-range researcher-selected bounds."""
    result: list[Diagnostic] = []
    for name in OPTIMIZABLE_PARAMETER_NAMES:
        variable = variables.get(name)
        if variable is None:
            result.append(_message("ERROR", "DESIGN SPACE", f"Missing variable entry: {name}."))
            continue
        lower = _number(variable.get("lower"))
        upper = _number(variable.get("upper"))
        if lower is None or upper is None:
            result.append(
                _message(
                    "ERROR",
                    "DESIGN SPACE",
                    f"{name} has a missing or non-finite Min/Max value. "
                    "Enter both numeric bounds.",
                )
            )
            continue
        if bool(variable.get("optimize", False)):
            if lower >= upper:
                result.append(
                    _message(
                        "ERROR",
                        "DESIGN SPACE",
                        f"{name} has zero or negative search width: "
                        f"min = {lower:g}, max = {upper:g}.\n"
                        "To fix: set Min < Max before running optimization.",
                    )
                )
            nominal_value = _number(nominal_parameters.get(name))
            if nominal_value is None:
                result.append(
                    _message(
                        "ERROR",
                        "DESIGN SPACE",
                        f"Nominal {name} is missing or non-finite.",
                    )
                )
            else:
                if lower > nominal_value:
                    result.append(
                        _message(
                            "ERROR",
                            "DESIGN SPACE",
                            f"{name} Min = {lower:g} is above nominal value "
                            f"{nominal_value:g}.\n"
                            f"To fix: set Min <= nominal value ({nominal_value:g}), "
                            "or change the nominal value.",
                        )
                    )
                if upper < nominal_value:
                    result.append(
                        _message(
                            "ERROR",
                            "DESIGN SPACE",
                            f"{name} Max = {upper:g} is below nominal value "
                            f"{nominal_value:g}.\n"
                            f"To fix: set Max >= nominal value ({nominal_value:g}), "
                            "or change the nominal value.",
                        )
                    )
    active_count = sum(
        bool(variables.get(name, {}).get("optimize", False))
        for name in OPTIMIZABLE_PARAMETER_NAMES
    )
    if active_count == 0:
        result.append(
            _message("WARN", "DESIGN SPACE", "No optimization variables are active.")
        )
    return tuple(result)


def build_led(values: Mapping[str, object]) -> LED:
    """Construct the current LED after callers have handled diagnostics."""
    payload = _merged(_LED_DEFAULTS, values)
    return LED(**payload)


def build_optical_parameters(values: Mapping[str, object]) -> OpticalParameters:
    """Construct optical parameters after callers have handled diagnostics."""
    return OpticalParameters(**_merged(_OPTICAL_DEFAULTS, values))


__all__ = [
    "Diagnostic",
    "DiagnosticSeverity",
    "build_led",
    "build_optical_parameters",
    "diagnose_design_space",
    "diagnose_geometry",
    "diagnose_led",
    "diagnose_optical",
    "diagnose_physical_state",
    "diagnose_state",
]

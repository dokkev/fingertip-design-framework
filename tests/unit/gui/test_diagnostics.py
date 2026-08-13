"""Focused pure-Python tests for actionable GUI diagnostics."""

from __future__ import annotations

from dataclasses import asdict

from model import LED
from gui.diagnostics import (
    diagnose_design_space,
    diagnose_geometry,
    diagnose_led,
    diagnose_mechanical,
    diagnose_optical,
)
from gui.baseline import current_lit_baseline


def _geometry() -> dict[str, object]:
    return asdict(current_lit_baseline())


def _text(diagnostics) -> str:
    return "\n".join(item.formatted for item in diagnostics)


def test_width_diagnostic_reports_current_values_and_numeric_relations() -> None:
    values = _geometry()
    values.update(flat_pad_width=12.0, stem_width=7.6)
    text = _text(diagnose_geometry(values))
    assert "flat_pad_width >" in text
    assert "stem_width <" in text
    assert "bond_extension_width=4" in text
    assert "flat_pad_width=12" in text


def test_ellipse_diagnostic_reports_penetration_and_correction_bounds() -> None:
    values = _geometry()
    values["void_height"] = 12.0
    text = _text(diagnose_geometry(values))
    assert "penetration=" in text
    assert "available ellipse depth=" in text
    assert "stem_height <" in text
    assert "semielliptical_pad_height >" in text


def test_led_fit_diagnostics_include_width_and_height_alternatives() -> None:
    values = _geometry()
    values["stem_width"] = 7.6
    values["stem_height"] = 6.0
    width_text = _text(diagnose_geometry(values, led=LED(width_mm=9.0)))
    height_text = _text(diagnose_geometry(values, led=LED(height_mm=9.0)))
    assert "LED width" in width_text
    assert "Increase stem_width or reduce LED width" in width_text
    assert "LED height" in height_text
    assert "Increase stem_height or reduce LED height" in height_text


def test_mechanical_diagnostics_report_model_supported_intervals() -> None:
    text = _text(diagnose_mechanical({"young_modulus_mpa": -1.0, "poisson_ratio": 0.5}))
    assert "young_modulus_mpa must be > 0 MPa" in text
    assert "-1 < poisson_ratio < 0.5" in text


def test_optical_diagnostics_report_nonnegative_and_anisotropy_intervals() -> None:
    text = _text(
        diagnose_optical(
            {"absorption_per_mm": -0.1, "anisotropy_g": 1.0}
        )
    )
    assert "absorption_per_mm must be >= 0" in text
    assert "-1 < anisotropy_g < 1" in text


def test_active_bounds_must_enclose_baseline_in_the_correct_direction() -> None:
    baseline = {name: 1.0 for name in (
        "flat_pad_width",
        "flat_pad_height",
        "semielliptical_pad_height",
        "stem_width",
        "stem_height",
        "void_width",
    )}
    variables = {
        name: {"optimize": False, "lower": 1.0, "upper": 1.0}
        for name in baseline
    }
    variables["stem_height"] = {
        "optimize": True,
        "lower": 2.0,
        "upper": 3.0,
    }
    text = _text(diagnose_design_space(baseline, variables))
    assert "stem_height Min = 2 is above baseline 1" in text
    variables["stem_height"] = {
        "optimize": True,
        "lower": -1.0,
        "upper": 0.0,
    }
    text = _text(diagnose_design_space(baseline, variables))
    assert "stem_height Max = 0 is below baseline 1" in text

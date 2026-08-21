"""Minimum-silicone-thickness geometry checks."""

from __future__ import annotations

import pytest

from lumo.finger import (
    FingertipParameters,
    InvalidFingertipParameters,
    PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM,
    silicone_thickness_measures,
    validate_minimum_silicone_thickness,
)


def test_nominal_thickness_is_finite_and_above_constraint() -> None:
    parameters = FingertipParameters(void_height=0.25)
    measures = silicone_thickness_measures(parameters)
    assert measures.side_ligament_mm > PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM
    assert measures.diagonal_ellipse_ligament_mm > PRODUCTION_MINIMUM_SILICONE_THICKNESS_MM
    assert measures.minimum_silicone_thickness_mm == pytest.approx(
        min(measures.side_ligament_mm, measures.diagonal_ellipse_ligament_mm)
    )
    assert validate_minimum_silicone_thickness(parameters) == measures


def test_thickness_is_independent_of_arc_resolution() -> None:
    coarse = FingertipParameters(void_height=0.25, arc_resolution=16)
    fine = FingertipParameters(void_height=0.25, arc_resolution=1024)
    assert silicone_thickness_measures(coarse) == silicone_thickness_measures(fine)


def test_aggressive_void_is_rejected_by_euclidean_thickness_gate() -> None:
    parameters = FingertipParameters(
        flat_pad_height=5.0,
        semielliptical_pad_height=9.0,
        stem_width=7.6,
        stem_height=7.0,
        void_width=7.0,
        void_height=2.0,
    )
    measures = silicone_thickness_measures(parameters)
    assert measures.minimum_silicone_thickness_mm < 2.0
    with pytest.raises(InvalidFingertipParameters, match="minimum silicone thickness"):
        validate_minimum_silicone_thickness(parameters)

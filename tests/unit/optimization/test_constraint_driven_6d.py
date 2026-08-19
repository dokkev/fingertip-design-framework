"""Focused constraints for the production constraint-driven six-D space."""

from __future__ import annotations

from math import cos, pi, sin

import numpy as np
import pytest
from shapely.geometry import LineString

from model import (
    FingertipParameters,
    InvalidFingertipParameters,
    silicone_thickness_measures,
    validate_minimum_silicone_thickness,
)
from model.fingertip_model import FingertipModel
from optimization.ax_adapter import AxSettings, create_ax_client
from optimization.design_space import (
    PRODUCTION_LINEAR_PARAMETER_CONSTRAINTS,
    PRODUCTION_SEARCH_BOUNDS,
)
from optimization.study import create_production_study


def _values(**updates: float) -> dict[str, float]:
    values = {
        "flat_pad_height": 5.0,
        "semielliptical_pad_height": 9.0,
        "stem_width": 7.6,
        "stem_height": 6.0,
        "void_width": 1.0,
        "void_height": 0.25,
    }
    values.update(updates)
    return values


def test_independent_heights_and_exact_total_depth_boundary() -> None:
    space = create_production_study().design_space
    decoded = space.decode(_values(flat_pad_height=15.0, semielliptical_pad_height=15.0))
    assert decoded.flat_pad_height == 15.0
    assert decoded.semielliptical_pad_height == 15.0
    assert decoded.total_pad_depth == 30.0

    with pytest.raises(InvalidFingertipParameters, match="total pad depth"):
        space.decode(_values(flat_pad_height=15.1, semielliptical_pad_height=15.0))


def test_exact_side_thickness_passes_and_below_threshold_fails() -> None:
    space = create_production_study().design_space
    exact = space.decode(
        _values(
            flat_pad_height=10.0,
            semielliptical_pad_height=10.0,
            stem_width=8.0,
            void_width=6.0,
            void_height=1.0,
        )
    )
    assert silicone_thickness_measures(exact).minimum_silicone_thickness_mm == pytest.approx(5.0)
    with pytest.raises(InvalidFingertipParameters, match="minimum silicone thickness"):
        space.decode(
            _values(
                flat_pad_height=10.0,
                semielliptical_pad_height=10.0,
                stem_width=9.0,
                void_width=6.0,
                void_height=1.0,
            )
        )


def test_ellipse_threshold_is_authoritative() -> None:
    passing = FingertipParameters(void_height=2.0)
    failing = FingertipParameters(void_height=2.5)
    assert silicone_thickness_measures(passing).minimum_silicone_thickness_mm >= 5.0
    assert silicone_thickness_measures(failing).minimum_silicone_thickness_mm < 5.0
    validate_minimum_silicone_thickness(passing)
    with pytest.raises(InvalidFingertipParameters):
        validate_minimum_silicone_thickness(failing)


def _independent_polyline_oracle(parameters: FingertipParameters) -> float:
    a = parameters.flat_pad_width / 2.0
    internal = [
        LineString(FingertipModel(parameters).cutout_geometry.boundary.coords)
    ]
    arc = LineString(
        [
            (a * cos(theta), -parameters.flat_pad_height - parameters.semielliptical_pad_height * sin(theta))
            for theta in np.linspace(0.0, pi, 20001)
        ]
    )
    outer = [
        LineString([(-a, parameters.bond_extension_height), (-a, -parameters.flat_pad_height)]),
        LineString([(a, -parameters.flat_pad_height), (a, parameters.bond_extension_height)]),
        arc,
    ]
    return min(left.distance(right) for left in internal for right in outer)


def test_global_dmin_matches_independent_high_resolution_oracle() -> None:
    parameters = FingertipParameters(
        flat_pad_height=8.0,
        semielliptical_pad_height=12.0,
        stem_width=8.0,
        stem_height=6.0,
        void_width=2.0,
        void_height=0.5,
    )
    measures = silicone_thickness_measures(parameters)
    oracle = _independent_polyline_oracle(parameters)
    assert abs(measures.minimum_silicone_thickness_mm - oracle) < 2.0e-4


def test_global_dmin_is_arc_resolution_independent() -> None:
    coarse = FingertipParameters(void_height=0.25, arc_resolution=16)
    fine = FingertipParameters(void_height=0.25, arc_resolution=2048)
    assert silicone_thickness_measures(coarse) == silicone_thickness_measures(fine)


@pytest.mark.parametrize(
    "values",
    (
        _values(flat_pad_height=2.0, semielliptical_pad_height=6.0, stem_width=6.0, stem_height=2.0, void_width=2.0, void_height=0.0),
        _values(flat_pad_height=15.0, semielliptical_pad_height=14.0, stem_width=6.0, stem_height=10.0, void_width=1.0, void_height=2.0),
    ),
)
def test_shallow_wide_and_deep_narrow_morphologies_are_valid(values) -> None:
    assert create_production_study().design_space.decode(values).total_pad_depth <= 30.0


def test_invalid_candidate_is_rejected_before_mesh(monkeypatch) -> None:
    from optimization import evaluator as evaluator_module

    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("mesh-owning Fingertip was called for invalid design")

    monkeypatch.setattr(evaluator_module, "Fingertip", forbidden)
    study = create_production_study()
    evaluation = study.create_evaluator().evaluate(
        FingertipParameters(
            flat_pad_height=10.0,
            semielliptical_pad_height=10.0,
            stem_width=9.0,
            stem_height=6.0,
            void_width=6.0,
            void_height=1.0,
        )
    )
    assert evaluation.status == "invalid_design"
    assert calls == []


def test_ax_has_six_active_variables_and_native_linear_constraints() -> None:
    study = create_production_study()
    client = create_ax_client(study, AxSettings(1, 0, seed=20260819))
    assert tuple(client._experiment.search_space.parameters) == tuple(
        name for name, _, _ in PRODUCTION_SEARCH_BOUNDS
    )
    assert tuple(str(item) for item in client._experiment.search_space.parameter_constraints) == (
        "ParameterConstraint(1.0*flat_pad_height + 1.0*semielliptical_pad_height <= 30.0)",
        "ParameterConstraint(1.0*stem_width + 2.0*void_width <= 20.0)",
    )
    assert PRODUCTION_LINEAR_PARAMETER_CONSTRAINTS

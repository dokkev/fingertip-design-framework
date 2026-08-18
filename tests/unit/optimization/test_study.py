"""Focused tests for fixed optimization-study configuration."""

from __future__ import annotations

import pytest

from mesh import mesh_settings_for_level
from model import FingertipParameters, LED, OpticalMaterial
from optics import TraceSettings
from optimization import (
    DesignSpace,
    DesignVariable,
    OptimizationStudy,
    ScenarioGrid,
    OPTIMIZABLE_PARAMETER_NAMES,
)
from optimization.evaluator import DesignEvaluator


def _design_space(
    *,
    active: tuple[str, ...] = ("stem_width",),
    lower: dict[str, float] | None = None,
    upper: dict[str, float] | None = None,
    nominal_parameters: FingertipParameters | None = None,
) -> DesignSpace:
    nominal_parameters = nominal_parameters or FingertipParameters()
    lower = lower or {}
    upper = upper or {}
    variables = tuple(
        DesignVariable(
            name,
            name in active,
            lower.get(
                name,
                0.0 if name == "void_width" and name in active else getattr(nominal_parameters, name) - 0.5,
            ),
            upper.get(
                name,
                1.0 if name == "void_width" and name in active else getattr(nominal_parameters, name) + 0.5,
            ),
        )
        for name in OPTIMIZABLE_PARAMETER_NAMES
    )
    return DesignSpace(nominal_parameters, variables)


def _grid(*, adjacent: bool = True) -> ScenarioGrid:
    return ScenarioGrid(
        locations_x_mm=(0.0, 1.0) if adjacent else (0.0,),
        indentations_mm=(0.5,),
        indenter_radii_mm=(2.0,),
    )


def _trace_settings() -> TraceSettings:
    return TraceSettings(
        ray_count=3,
        grid_width=16,
        grid_height=16,
        maximum_segment_count=32,
    )


def _study(**overrides) -> OptimizationStudy:
    values = {
        "design_space": _design_space(),
        "scenario_grid": _grid(),
        "mesh_settings": mesh_settings_for_level("medium"),
        "trace_settings": _trace_settings(),
        "led": LED(),
        "optical": OpticalMaterial(),
    }
    values.update(overrides)
    return OptimizationStudy(**values)


def test_study_is_immutable_and_contains_fixed_scientific_configuration() -> None:
    led = LED(width_mm=3.0, height_mm=1.5, relative_radiant_power=0.8)
    optical = OpticalMaterial(absorption_per_mm=0.04, anisotropy_g=0.2)
    study = _study(
        led=led,
        optical=optical,
        fem_steps=9,
        internal_contact="three_pairs",
        basal_interface="explicit_contact",
    )

    assert study.design_space.active_variables[0].name == "stem_width"
    assert study.scenario_grid == _grid()
    assert study.mesh_settings.level == "medium"
    assert study.trace_settings.ray_count == 3
    assert study.led is led
    assert study.optical is optical
    assert study.fem_steps == 9
    assert study.internal_contact == "three_pairs"
    assert study.basal_interface == "explicit_contact"
    with pytest.raises(AttributeError):
        study.fem_steps = 10  # type: ignore[misc]


def test_production_study_defaults_to_validated_12_step_search() -> None:
    study = _study()
    assert study.fem_steps == 12
    assert study.internal_contact == "sides_separate"


def test_validation_reference_can_explicitly_keep_48_steps() -> None:
    assert _study(fem_steps=48).fem_steps == 48


def test_study_rejects_zero_active_variables() -> None:
    with pytest.raises(ValueError, match="at least one active"):
        _study(design_space=_design_space(active=()))


def test_study_rejects_zero_width_active_variable() -> None:
    with pytest.raises(ValueError, match="zero search width"):
        _study(
            design_space=_design_space(
                lower={"stem_width": 7.6},
                upper={"stem_width": 7.6},
            )
        )


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    (
        ({"stem_width": 7.7}, {"stem_width": 8.0}, "outside"),
        ({"stem_width": 6.0}, {"stem_width": 7.5}, "outside"),
    ),
)
def test_study_requires_nominal_inside_active_bounds(lower, upper, message) -> None:
    with pytest.raises(ValueError, match=f"nominal stem_width=7.6.*{message}"):
        _study(design_space=_design_space(lower=lower, upper=upper))


def test_study_rejects_missing_adjacent_pair_and_invalid_fem_steps() -> None:
    with pytest.raises(ValueError, match="adjacent"):
        _study(scenario_grid=_grid(adjacent=False))
    with pytest.raises(ValueError, match="positive integer"):
        _study(fem_steps=0)
    with pytest.raises(ValueError, match="positive integer"):
        _study(fem_steps=True)
    with pytest.raises(ValueError, match="non-empty"):
        _study(internal_contact="")


def test_study_allows_infeasible_box_corner_without_checking_or_repairing_it() -> None:
    design_space = _design_space(
        active=("flat_pad_width", "stem_width"),
        lower={"flat_pad_width": 15.0, "stem_width": 7.6},
        upper={"flat_pad_width": 20.0, "stem_width": 9.0},
        nominal_parameters=FingertipParameters(flat_pad_width=20.0),
    )
    study = _study(design_space=design_space)
    assert study.design_space is design_space
    assert design_space.corner_values()[-1] == {
        "flat_pad_width": 20.0,
        "stem_width": 9.0,
    }


def test_study_validates_fixed_led_fit_through_public_fingertip() -> None:
    with pytest.raises(ValueError, match="LED package width"):
        _study(led=LED(width_mm=8.0))


def test_create_evaluator_binds_exact_study_configuration() -> None:
    study = _study()
    evaluator = study.create_evaluator()

    assert isinstance(evaluator, DesignEvaluator)
    assert evaluator.scenario_grid is study.scenario_grid
    assert evaluator.mesh_settings is study.mesh_settings
    assert evaluator.trace_settings is study.trace_settings
    assert evaluator.led is study.led
    assert evaluator.optical is study.optical
    assert evaluator.fem_steps == study.fem_steps
    assert evaluator.internal_contact == study.internal_contact


def test_imported_study_construction_does_not_require_solver_execution() -> None:
    _study()

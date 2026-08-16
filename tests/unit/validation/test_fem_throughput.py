from __future__ import annotations

import numpy as np

from fem.kratos_settings import (
    DEFAULT_INDENTATION_SOLVER_SETTINGS,
    IndentationSolverSettings,
    build_indentation_project_parameters_data,
)
from validation.fem.throughput import (
    STEP_DECISION_COUNTS,
    _step_decision_index,
    _step_decision_ordering,
    _mesh_policies,
    _profile_error,
    _read_morphologies,
)


def test_mesh_policy_sweep_keeps_contact_refinement_bounded() -> None:
    policies = {policy.name: policy for policy in _mesh_policies()}

    assert policies["reference_medium"].settings.bulk_target_size_mm == 0.75
    assert policies["coarse_b"].settings.contact_boundary_target_size_mm == 0.40
    assert policies["coarse_c"].settings.contact_refinement_distance_mm == 0.75
    assert all(
        policy.settings.contact_boundary_target_size_mm
        <= policy.settings.bulk_target_size_mm
        for policy in policies.values()
    )


def test_boundary_comparison_is_independent_of_node_sampling() -> None:
    reference_u = np.linspace(0.0, 1.0, 11)
    candidate_u = np.linspace(0.0, 1.0, 23)
    reference = np.column_stack((reference_u, 2.0 * reference_u + 1.0, -reference_u))
    candidate = np.column_stack((candidate_u, 2.0 * candidate_u + 1.0, -candidate_u))

    error = _profile_error(candidate, reference)

    assert error["rms_position_error_mm"] < 1.0e-12
    assert error["maximum_position_error_mm"] < 1.0e-12


def test_solver_override_does_not_change_trusted_defaults() -> None:
    trusted = build_indentation_project_parameters_data(48)
    fast = build_indentation_project_parameters_data(
        16,
        solver_settings=IndentationSolverSettings(
            relative_tolerance=1.0e-5,
            absolute_tolerance=1.0e-8,
            reform_dofs_at_each_step=False,
        ),
    )

    assert DEFAULT_INDENTATION_SOLVER_SETTINGS.relative_tolerance == 1.0e-6
    assert trusted["solver_settings"]["displacement_relative_tolerance"] == 1.0e-6
    assert trusted["solver_settings"]["reform_dofs_at_each_step"] is True
    assert fast["solver_settings"]["displacement_relative_tolerance"] == 1.0e-5
    assert fast["solver_settings"]["reform_dofs_at_each_step"] is False


def test_required_morphologies_keep_nominal_and_candidate_separate() -> None:
    morphologies = {item.name: item for item in _read_morphologies()}

    assert morphologies["nominal"].source == "FingertipParameters()"
    assert morphologies["nominal"].parameters != morphologies["candidate49"].parameters
    assert morphologies["difficult_candidate50"].source.endswith(
        "candidate_0050.json"
    )


def test_step_decision_uses_only_requested_step_counts() -> None:
    assert STEP_DECISION_COUNTS == (48, 24, 12)


def test_step_decision_indexes_continuation_snapshots_by_depth_and_location() -> None:
    records = [{
        "morphology": "nominal",
        "requested_steps": 12,
        "scenario": {"location_x_mm": -3.0, "indentation_mm": 1.0},
        "snapshots": {"0.5": {"step": 6}, "1": {"step": 12}},
    }]

    indexed = _step_decision_index(records)

    assert indexed[("nominal", 12, -3.0, 0.5)][1]["step"] == 6
    assert indexed[("nominal", 12, -3.0, 1.0)][1]["step"] == 12


def test_step_decision_reports_candidate_ordering_per_depth() -> None:
    pair_metrics = [
        {"morphology": "nominal", "steps": 12, "indentation_mm": 0.5, "separability": 0.07},
        {"morphology": "candidate49", "steps": 12, "indentation_mm": 0.5, "separability": 0.12},
        {"morphology": "nominal", "steps": 12, "indentation_mm": 1.0, "separability": 0.10},
        {"morphology": "candidate49", "steps": 12, "indentation_mm": 1.0, "separability": 0.19},
    ]

    checks = _step_decision_ordering(pair_metrics, 12)

    assert all(check["candidate49_above_nominal"] for check in checks)

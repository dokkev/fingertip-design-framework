from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import lumo.simulation as simulation_module
from lumo import MechanicsContract
from lumo.simulation import LumoSimulation
from physics import CandidateMechanicsError


def test_checkpoint_values_are_absolute_depths_with_derived_annotations() -> None:
    fractions, ratios = LumoSimulation._checkpoint_values((0.5, 1.0, 1.5), 5.0)

    assert fractions == (1.0 / 3.0, 2.0 / 3.0, 1.0)
    assert ratios == (0.1, 0.2, 0.3)


def test_optix_runtime_is_created_once_and_reused(monkeypatch) -> None:
    simulation = object.__new__(LumoSimulation)
    simulation.optix_runtime = None
    created: list[object] = []
    runtime = object()

    def create_runtime():
        created.append(runtime)
        return runtime

    monkeypatch.setattr(simulation_module, "create_runtime", create_runtime)

    assert simulation._runtime() is runtime
    assert simulation._runtime() is runtime
    assert created == [runtime]


def test_checkpoint_values_reject_non_monotonic_or_non_finite_depths() -> None:
    with np.testing.assert_raises(ValueError):
        LumoSimulation._checkpoint_values((0.5, 0.5), 5.0)
    with np.testing.assert_raises(ValueError):
        LumoSimulation._checkpoint_values((0.5, float("nan")), 5.0)


def test_mechanics_contract_rejects_non_finite_or_non_integer_settings() -> None:
    with np.testing.assert_raises(ValueError):
        MechanicsContract(dt_s=float("nan"))
    with np.testing.assert_raises(ValueError):
        MechanicsContract(soft_contact_ke=float("inf"))
    with np.testing.assert_raises(TypeError):
        MechanicsContract(vbd_iterations=10.0)  # type: ignore[arg-type]
    with np.testing.assert_raises(TypeError):
        MechanicsContract(sphere_subdivisions=True)  # type: ignore[arg-type]


def _acceptance_subject(
    diagnostics: dict[str, object],
    *,
    final_pose_error_mm: float = 0.0,
) -> tuple[LumoSimulation, SimpleNamespace]:
    simulation = object.__new__(LumoSimulation)
    simulation.mechanics_contract = MechanicsContract()
    diagnostics = {
        "final_pose_error_mm": final_pose_error_mm,
        **diagnostics,
    }
    checkpoint = SimpleNamespace(
        diagnostics=diagnostics,
        post_contact_travel_mm=1.0,
    )
    return simulation, checkpoint


def test_checkpoint_acceptance_allows_a_state_at_the_explicit_limits() -> None:
    simulation, checkpoint = _acceptance_subject(
        {
            "inverted_tetrahedra": 0,
            "max_soft_contact_overflow": 0,
            "max_rigid_contact_overflow": 0,
            "max_support_displacement_mm": 0.0,
            "carrier_collision_enabled": True,
            "rigid_sdf_target_voxel_mm": 0.125,
            "max_carrier_penetration_mm": 0.0625,
        }
    )

    simulation._validate_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"inverted_tetrahedra": 1}, "inverted tetrahedra"),
        ({"max_soft_contact_overflow": 1}, "contact buffer overflow"),
        ({"max_rigid_contact_overflow": 1}, "contact buffer overflow"),
        ({"max_support_displacement_mm": 1.0e-3}, "support displacement"),
        ({"max_carrier_penetration_mm": 0.0626}, "carrier penetration"),
    ),
)
def test_checkpoint_acceptance_rejects_candidate_mechanics_failures(
    overrides: dict[str, object],
    message: str,
) -> None:
    diagnostics: dict[str, object] = {
        "inverted_tetrahedra": 0,
        "max_soft_contact_overflow": 0,
        "max_rigid_contact_overflow": 0,
        "max_support_displacement_mm": 0.0,
        "carrier_collision_enabled": True,
        "rigid_sdf_target_voxel_mm": 0.125,
        "max_carrier_penetration_mm": 0.0,
    }
    diagnostics.update(overrides)
    simulation, checkpoint = _acceptance_subject(diagnostics)

    with pytest.raises(CandidateMechanicsError, match=message):
        simulation._validate_checkpoint(checkpoint)


def test_checkpoint_acceptance_rejects_prescribed_pose_error() -> None:
    simulation, checkpoint = _acceptance_subject(
        {
            "inverted_tetrahedra": 0,
            "max_soft_contact_overflow": 0,
            "max_rigid_contact_overflow": 0,
            "max_support_displacement_mm": 0.0,
            "carrier_collision_enabled": False,
            "rigid_sdf_target_voxel_mm": 0.125,
        },
        final_pose_error_mm=0.1,
    )

    with pytest.raises(CandidateMechanicsError, match="prescribed-pose error"):
        simulation._validate_checkpoint(checkpoint)


def test_checkpoint_acceptance_does_not_classify_invalid_static_settings_as_candidate_failure() -> None:
    simulation, checkpoint = _acceptance_subject(
        {
            "inverted_tetrahedra": 0,
            "max_soft_contact_overflow": 0,
            "max_rigid_contact_overflow": 0,
            "max_support_displacement_mm": 0.0,
            "carrier_collision_enabled": False,
            "rigid_sdf_target_voxel_mm": float("nan"),
        }
    )

    with pytest.raises(RuntimeError, match="voxel size"):
        simulation._validate_checkpoint(checkpoint)


def test_checkpoint_acceptance_fails_closed_when_evidence_is_missing() -> None:
    simulation = object.__new__(LumoSimulation)
    simulation.mechanics_contract = MechanicsContract()
    checkpoint = SimpleNamespace(diagnostics={}, post_contact_travel_mm=1.0)

    with pytest.raises(RuntimeError, match="required diagnostic"):
        simulation._validate_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("indices", "message"),
    (
        ((0, 0), "duplicate"),
        ((1,), "out-of-range"),
        ((0.5,), "1D integer"),
    ),
)
def test_carrier_contact_provenance_rejects_malformed_local_indices(
    indices: tuple[object, ...],
    message: str,
) -> None:
    simulation = object.__new__(LumoSimulation)
    simulation.prepared = SimpleNamespace(
        source_node_ids=np.asarray([100], dtype=np.int64)
    )
    checkpoint = SimpleNamespace(
        diagnostics={"active_carrier_contact_vertex_indices": indices}
    )

    with pytest.raises(RuntimeError, match=message):
        simulation._carrier_contact_source_ids(checkpoint)

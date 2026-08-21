from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import validation.optimization.lumo3d_scientific_convergence as convergence
from lumo.config import load_lumo_execution_config
from lumo.finger import FingertipParameters
from lumo.optimization.design_space import (
    DesignSpace,
    DesignVariable,
    PRODUCTION_LINEAR_CONSTRAINTS,
    PRODUCTION_SEARCH_BOUNDS,
)
from validation.common.io import atomic_write_json
from validation.optimization.lumo3d_scientific_convergence import (
    NEWTON_RELATIVE_MAX_THRESHOLD,
    NEWTON_RMS_THRESHOLD_MM,
    compare_mesh_evaluations,
    compare_newton_evaluations,
    convergence_plan,
    optical_sweep_settings,
)
from validation.optimization.representative_morphologies import (
    representative_morphologies,
)


ROOT = Path(__file__).resolve().parents[3]


def _design_space() -> DesignSpace:
    return DesignSpace(
        FingertipParameters(void_height=0.25),
        tuple(
            DesignVariable(spec.name, True, spec.lower, spec.upper)
            for spec in PRODUCTION_SEARCH_BOUNDS
        ),
        linear_constraints=PRODUCTION_LINEAR_CONSTRAINTS,
    )


def test_representative_morphologies_are_ordered_unique_and_feasible() -> None:
    design_space = _design_space()
    cases = representative_morphologies(design_space)

    assert [case.case_id for case in cases] == [
        "nominal",
        "latent_center",
        "wide_cutout_edge",
        "deep_cutout_edge",
        "minimum_wall_edge",
    ]
    assert len({case.morphology_fingerprint for case in cases}) == 5
    for case in cases:
        assert all(0.0 <= value <= 1.0 for value in case.latent_values.values())
        design_space.validate_physical_parameters(case.parameters)
        assert case.to_dict()["latent_values"] == dict(case.latent_values)
    minimum_wall = cases[-1].thickness_measures.minimum_silicone_thickness_mm
    assert 5.0 <= minimum_wall <= 5.5


def test_convergence_plan_preserves_thresholds_and_unsupported_force() -> None:
    execution = load_lumo_execution_config(ROOT / "config" / "lumo_execution.yaml")
    plan = convergence_plan(execution)

    assert plan["newton"]["production"]["vbd_iterations"] == 10
    assert plan["newton"]["reference"]["vbd_iterations"] == 20
    assert plan["newton"]["reference"]["max_load_increment_mm"] <= 0.025
    assert plan["newton"]["acceptance"] == {
        "rms_vertex_difference_mm_max": NEWTON_RMS_THRESHOLD_MM,
        "relative_max_displacement_difference_max": NEWTON_RELATIVE_MAX_THRESHOLD,
    }
    assert plan["mesh"]["force_metric"]["status"] == "unsupported"
    assert plan["mesh"]["force_metric"]["value_n"] is None
    assert plan["mesh"]["scientific_threshold"] is None
    assert plan["optics"]["family_reference_pairs"]["ray_count"] == {
        "production_setting_id": "production",
        "reference_setting_id": "rays_1024",
    }


def test_each_optical_sweep_changes_only_its_declared_family() -> None:
    execution = load_lumo_execution_config(ROOT / "config" / "lumo_execution.yaml")
    specs = optical_sweep_settings(execution)
    baseline = asdict(specs[0]["settings"])
    allowed = {
        "ray_count": {"ray_count"},
        "max_interactions": {"max_interactions"},
        "maximum_segment_count": {"maximum_segment_count"},
        "path_field_grid": {
            "internal_grid_width",
            "internal_grid_height",
            "internal_z_bins",
        },
    }

    assert specs[0]["family"] == "baseline"
    assert specs[0]["role"] == "production"
    for spec in specs[1:]:
        resolved = asdict(spec["settings"])
        changed = {name for name in baseline if baseline[name] != resolved[name]}
        assert changed == allowed[spec["family"]]
    for family in allowed:
        family_specs = [spec for spec in specs if spec["family"] == family]
        assert [spec["role"] for spec in family_specs] == [
            "intermediate",
            "reference",
        ]


def test_mesh_candidate_failure_is_preserved_separately_from_objective() -> None:
    failed = SimpleNamespace(
        status="mesh_failure",
        objective_value=None,
        objective=None,
        report={},
        failure_message="quality gate",
        failure_scenario=None,
        result_artifact_path="failure.json",
        checkpoint_records=(),
    )
    result = compare_mesh_evaluations(failed, failed)

    assert result["execution_status"] == "mesh_failure"
    assert result["scientific_convergence"] == "FAIL"
    assert result["search"]["objective"] is None


def test_newton_comparison_uses_complete_mechanics_despite_optics_failure(
    monkeypatch,
) -> None:
    record = SimpleNamespace(trajectory_id="trajectory", checkpoint_index=0)
    failed_optics = SimpleNamespace(
        status="optics_failure",
        objective_value=None,
        objective=None,
        report={"failure_stage": "objective"},
        failure_message="objective pathology",
        failure_scenario="objective_pathology",
        result_artifact_path="failure.json",
        checkpoint_records=(record,),
    )
    monkeypatch.setattr(
        convergence,
        "_mechanics_checkpoint_evidence",
        lambda _evaluation: {"complete": True, "status": "completed"},
    )
    monkeypatch.setattr(
        convergence,
        "_evaluation_record",
        lambda evaluation: {"evaluation_status": evaluation.status},
    )
    arrays = {
        "rest_vertices_mm": np.zeros((2, 3)),
        "deformed_vertices_mm": np.full((2, 3), 0.001),
        "tetrahedra": np.asarray([[0, 1, 1, 0]]),
        "source_node_ids": np.asarray([0]),
    }
    monkeypatch.setattr(convergence, "_load_mechanics_arrays", lambda _path: arrays)
    record.mechanics_artifact_path = Path("unused.npz")

    result = compare_newton_evaluations(failed_optics, failed_optics)

    assert result["execution_status"] == "completed"
    assert result["scientific_convergence"] == "PASS"
    assert result["objective_sensitivity"]["objective"] is None


def test_mesh_comparison_preserves_completed_mesh_and_mechanics_on_optics_failure(
    monkeypatch,
) -> None:
    failed_optics = SimpleNamespace(
        status="optics_failure",
        objective_value=None,
        objective=None,
        report={"volume_mesh": {"quality": {"minimum": 0.2}}},
        failure_message="optical acceptance",
        failure_scenario="numerical_acceptance",
        result_artifact_path="failure.json",
        checkpoint_records=(),
    )
    monkeypatch.setattr(
        convergence,
        "_mechanics_checkpoint_evidence",
        lambda _evaluation: {"complete": True, "status": "completed"},
    )
    monkeypatch.setattr(
        convergence,
        "_displacement_scalars",
        lambda _evaluation: {
            ("trajectory", 0): {
                "maximum_displacement_mm": 1.0,
                "rms_displacement_mm": 0.5,
            }
        },
    )

    result = compare_mesh_evaluations(failed_optics, failed_optics)

    assert result["execution_status"] == "completed"
    assert result["scientific_convergence"] == "INCONCLUSIVE"
    assert result["objective_sensitivity"]["objective"] is None
    assert result["mesh_statistics"]["reference"] == failed_optics.report["volume_mesh"]


def test_validation_json_writer_rejects_nonfinite_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        atomic_write_json(tmp_path / "invalid.json", {"value": float("nan")})


def test_optical_acceptance_and_sensitivity_are_separate_and_machine_readable() -> None:
    upstream_failure = SimpleNamespace(
        status="mechanics_failure",
        failure_scenario="candidate_mechanics_state",
    )
    hard_optical_failure = SimpleNamespace(
        status="optics_failure",
        failure_scenario="numerical_acceptance",
    )
    assert convergence._baseline_numerical_acceptance(upstream_failure) == "NOT_RUN"
    assert convergence._baseline_numerical_acceptance(hard_optical_failure) == "FAIL"

    sensitivity = convergence._optical_sensitivity(
        {
            "objective": None,
            "D_inter": None,
            "D_radius": None,
            "diagnostic_raw_objective": {
                "accepted": False,
                "objective": 2.0,
                "D_inter": 4.0,
                "D_radius": 1.0,
            },
        },
        {
            "setting_id": "rays_1024",
            "family": "ray_count",
            "role": "reference",
            "objective": 3.0,
            "D_inter": 2.0,
            "D_radius": 1.5,
        },
    )
    assert sensitivity["scientific_threshold"] is None
    assert sensitivity["scientific_convergence"] == "INCONCLUSIVE"
    assert sensitivity["metrics"]["objective"]["signed_delta"] == pytest.approx(1.0)
    assert sensitivity["metrics"]["D_inter"]["absolute_delta"] == pytest.approx(2.0)


def test_optical_replay_uses_production_carrier_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    execution = load_lumo_execution_config(ROOT / "config" / "lumo_execution.yaml")
    case = representative_morphologies(_design_space())[0]
    record = SimpleNamespace(
        mechanics_artifact_path=tmp_path / "state.npz",
        mechanics_artifact_sha256="digest",
        trajectory_id="trajectory",
        checkpoint_index=0,
        normalized_location=0.5,
        radius_mm=5.0,
        checkpoint_depth_mm=1.0,
        contact_state=SimpleNamespace(to_dict=lambda: {"state": "test"}),
    )
    production = SimpleNamespace(status="optics_failure", checkpoint_records=(record,))
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        convergence,
        "generate_volume_mesh",
        lambda *_args: SimpleNamespace(solid=object()),
    )
    monkeypatch.setattr(convergence, "prepare_fingertip_mesh", lambda *_args: object())
    monkeypatch.setattr(convergence, "make_distal_phalanx_mesh", lambda *_args: object())
    monkeypatch.setattr(convergence, "create_runtime", lambda *_args: object())
    monkeypatch.setattr(
        convergence,
        "_mechanics_checkpoint_evidence",
        lambda _evaluation: {"complete": True, "status": "completed"},
    )

    def restore(*_args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(geometry=object())

    monkeypatch.setattr(convergence, "restore_deformed_optical_state", restore)
    monkeypatch.setattr(
        convergence,
        "trace_geometry",
        lambda *_args, **_kwargs: SimpleNamespace(
            field=np.ones((2, 2), dtype=float),
            total_transport=1.0,
            escaped_weight=1.0,
        ),
    )
    monkeypatch.setattr(convergence, "save_case_artifact", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(convergence, "energy_record", lambda *_args: {})
    monkeypatch.setattr(
        convergence,
        "DEFAULT_OPTICAL_NUMERICAL_ACCEPTANCE",
        SimpleNamespace(
            assess=lambda _result: SimpleNamespace(
                accepted=True,
                to_dict=lambda: {"accepted": True},
            )
        ),
    )
    monkeypatch.setattr(
        convergence,
        "compute_trajectory_objective",
        lambda *_args, **_kwargs: SimpleNamespace(
            objective_pathology=False,
            objective_value=1.0,
            d_inter=2.0,
            d_radius=3.0,
        ),
    )

    result = convergence._replay_optics(
        tmp_path / "replay",
        case,
        production,
        execution,
        "production",
        execution.transport,
    )

    assert result["numerical_acceptance"] == "PASS"
    assert result["baseline_evaluation_status"] == "optics_failure"
    assert seen["carrier_optics"].boundary_model == "absorber"
    assert seen["carrier_mapping_tolerance_mm"] == pytest.approx(
        0.5 * execution.mechanics.rigid_sdf_target_voxel_mm
    )
    assert seen["source_epsilon_mm"] == pytest.approx(
        execution.transport.source_epsilon_mm
    )

    def fail_trace(*_args, **_kwargs):
        raise convergence.Transport3DCandidateGeometryError(
            "outside observation grid"
        )

    monkeypatch.setattr(convergence, "trace_geometry", fail_trace)
    failed = convergence._replay_optics(
        tmp_path / "candidate-failure",
        case,
        production,
        execution,
        "production",
        execution.transport,
    )
    assert failed["execution_status"] == "candidate_failure"
    assert failed["cause_type"] == "Transport3DCandidateGeometryError"
    assert failed["numerical_acceptance"] == "NOT_RUN"


def test_convergence_wrapper_persists_infrastructure_or_invariant_abort(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "convergence"

    def fail(root, **_kwargs):
        Path(root).mkdir(parents=True)
        atomic_write_json(Path(root) / "config.json", {"schema": "test"})
        raise RuntimeError("fatal invariant")

    monkeypatch.setattr(convergence, "_execute_scientific_convergence", fail)

    with pytest.raises(RuntimeError, match="fatal invariant"):
        convergence.run_scientific_convergence(output)

    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "ERROR"
    assert summary["execution_status"] == "infrastructure_or_invariant_failure"


def test_convergence_rejects_unavailable_source_before_expensive_evaluation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        convergence,
        "_source_provenance",
        lambda: {"status": "unavailable", "source_id": None},
    )
    monkeypatch.setattr(
        convergence,
        "Lumo3DTrajectoryEvaluator",
        lambda *_args, **_kwargs: pytest.fail("expensive evaluator was constructed"),
    )

    with pytest.raises(RuntimeError, match="provenance is unavailable"):
        convergence._execute_scientific_convergence(tmp_path / "unavailable")
    assert not (tmp_path / "unavailable").exists()

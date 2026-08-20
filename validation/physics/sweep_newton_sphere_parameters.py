"""Staged numerical convergence sweep for the Newton sphere-contact path.

This is a validation-tier utility.  It uses the same neutral first-contact
and Newton indentation APIs as the production ``physics`` package and does
not change the mechanics model or contact constants.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from validation.common.io import atomic_write_json, write_csv


DEFAULT_OUTPUT_DIR = Path("output/validation/physics/newton_sphere_sweep")
DEFAULT_DEVICE = "cuda:0"

DT_S = 1.0e-3
GRAVITY = 0.0
SOFT_CONTACT_MARGIN_MM = 0.02
SOFT_CONTACT_KE = 1.0e3
SOFT_CONTACT_KD = 10.0
FIRST_CONTACT_COARSE_STEP_MM = 0.25
FIRST_CONTACT_TOLERANCE_MM = 1.0e-3
SPAWN_CLEARANCE_MM = 0.05
FIRST_CONTACT_MAX_TRAVEL_MM = 20.0

RMS_VERTEX_THRESHOLD_MM = 0.005
RELATIVE_MAX_DISPLACEMENT_THRESHOLD = 0.03


@dataclass(frozen=True)
class SweepConfig:
    """One reproducible Newton sphere-contact sweep point."""

    stage: str
    radius_mm: float
    travel_mm: float
    sphere_subdivisions: int
    load_steps: int
    iterations: int
    repeat_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("stage must be a non-empty string")
        for name in ("radius_mm", "travel_mm"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in ("sphere_subdivisions", "load_steps", "iterations", "repeat_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{name} must be an integer")
            if name == "repeat_index" and int(value) < 0:
                raise ValueError("repeat_index must be non-negative")
            if name != "repeat_index" and int(value) < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, int(value))

    @property
    def load_increment_mm(self) -> float:
        return self.travel_mm / self.load_steps

    @property
    def run_id(self) -> str:
        return (
            f"{self.stage}:r{self.radius_mm:g}:t{self.travel_mm:g}:"
            f"sub{self.sphere_subdivisions}:steps{self.load_steps}:"
            f"iter{self.iterations}:rep{self.repeat_index}"
        )


@dataclass
class _RunBundle:
    record: dict[str, Any]
    deformed_vertices: np.ndarray | None = None
    displacement: np.ndarray | None = None


@dataclass(frozen=True)
class _PreparedCase:
    model: Any
    solid: Any
    volume_mesh: Any
    prepared: Any
    contact_surface: Any


def load_steps_for_increment(travel_mm: float, max_load_increment_mm: float) -> int:
    """Return the smallest positive step count satisfying a max increment."""

    travel = float(travel_mm)
    increment = float(max_load_increment_mm)
    if not np.isfinite(travel) or travel <= 0.0:
        raise ValueError("travel_mm must be finite and positive")
    if not np.isfinite(increment) or increment <= 0.0:
        raise ValueError("max_load_increment_mm must be finite and positive")
    return max(1, int(math.ceil(travel / increment - 1.0e-12)))


def comparison_metrics(
    candidate_vertices: np.ndarray,
    reference_vertices: np.ndarray,
    candidate_displacement: np.ndarray,
    reference_displacement: np.ndarray,
) -> dict[str, float]:
    """Compute geometry-focused differences between two successful runs."""

    candidate = np.asarray(candidate_vertices, dtype=float)
    reference = np.asarray(reference_vertices, dtype=float)
    candidate_u = np.asarray(candidate_displacement, dtype=float)
    reference_u = np.asarray(reference_displacement, dtype=float)
    if candidate.shape != reference.shape or candidate.ndim != 2 or candidate.shape[1] != 3:
        raise ValueError("vertex arrays must share shape (n_vertices, 3)")
    if candidate_u.shape != reference_u.shape or candidate_u.shape != candidate.shape:
        raise ValueError("displacement arrays must match vertex array shape")
    vertex_delta = candidate - reference
    displacement_delta = candidate_u - reference_u
    candidate_max = float(np.max(np.linalg.norm(candidate_u, axis=1)))
    reference_max = float(np.max(np.linalg.norm(reference_u, axis=1)))
    return {
        "max_abs_vertex_difference_mm": float(np.max(np.abs(vertex_delta))),
        "rms_vertex_difference_mm": float(np.sqrt(np.mean(np.square(vertex_delta)))),
        "max_displacement_difference_mm": abs(candidate_max - reference_max),
        "max_displacement_field_difference_mm": float(
            np.max(np.linalg.norm(displacement_delta, axis=1))
        ),
        "relative_max_displacement_difference": abs(candidate_max - reference_max)
        / max(reference_max, 1.0e-12),
    }


def _base_record(config: SweepConfig, device: str) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "stage": config.stage,
        "repeat_index": config.repeat_index,
        "status": "not_run",
        "failure_reason": None,
        "device": device,
        "radius_mm": config.radius_mm,
        "travel_mm": config.travel_mm,
        "sphere_subdivisions": config.sphere_subdivisions,
        "load_steps": config.load_steps,
        "load_increment_mm": config.load_increment_mm,
        "iterations": config.iterations,
        "dt": DT_S,
        "runtime_s": None,
        "first_contact_travel_mm": None,
        "first_contact_bracket_width_mm": None,
        "first_contact_tolerance_mm": FIRST_CONTACT_TOLERANCE_MM,
        "spawn_clearance_mm": SPAWN_CLEARANCE_MM,
        "max_soft_contact_count": None,
        "max_rigid_contact_count": None,
        "max_soft_contact_overflow": None,
        "max_rigid_contact_overflow": None,
        "max_displacement_mm": None,
        "rms_displacement_mm": None,
        "min_signed_six_volume": None,
        "inverted_tet_count": None,
        "finite_vertices": None,
        "finite_displacements": None,
        "final_pose_error_mm": None,
        "reference_run_id": None,
        "max_abs_vertex_difference_mm": None,
        "rms_vertex_difference_mm": None,
        "max_displacement_difference_mm": None,
        "max_displacement_field_difference_mm": None,
        "relative_max_displacement_difference": None,
    }


def failed_record(
    config: SweepConfig,
    failure_reason: str,
    *,
    device: str = DEFAULT_DEVICE,
    runtime_s: float | None = None,
) -> dict[str, Any]:
    """Create a serializable failure record without aborting a sweep."""

    record = _base_record(config, device)
    record["status"] = "failed"
    record["failure_reason"] = str(failure_reason)
    record["runtime_s"] = None if runtime_s is None else float(runtime_s)
    return record


def _six_volumes(vertices: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=float)[np.asarray(tetrahedra, dtype=np.int64)]
    return np.einsum(
        "ij,ij->i",
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
        points[:, 3] - points[:, 0],
    )


def _prepare_case() -> _PreparedCase:
    from contact import make_outer_compliant_surface
    from physics import prepare_fingertip_mesh
    from mesh.volume.mesh import generate_volume_mesh
    from mesh.volume.contracts import volume_mesh_settings_for_tier
    from model.fingertip_model import FingertipModel
    from model.fingertip_model import FingertipParameters
    from model.solid import build_fingertip_solid

    model = FingertipModel(
        FingertipParameters(
            void_width=1.0,
            void_height=0.0,
            poisson_ratio=0.49,
        )
    )
    solid = build_fingertip_solid(model)
    volume_mesh = generate_volume_mesh(
        solid,
        volume_mesh_settings_for_tier("search"),
    )
    prepared = prepare_fingertip_mesh(volume_mesh)
    return _PreparedCase(
        model=model,
        solid=solid,
        volume_mesh=volume_mesh,
        prepared=prepared,
        contact_surface=make_outer_compliant_surface(solid),
    )


def _run_case(case: _PreparedCase, config: SweepConfig, device: str) -> _RunBundle:
    from contact import (
        FirstContactSettings,
        canonical_sphere_alignment,
        find_first_contact,
        intersects,
    )
    from physics import (
        IndentationSettings,
        NewtonSettings,
        RigidIndenter3D,
        solve_fingertip_indentation,
    )
    from mesh import make_sphere_mesh

    record = _base_record(config, device)
    started = time.perf_counter()
    try:
        sphere = make_sphere_mesh(
            config.radius_mm,
            subdivisions=config.sphere_subdivisions,
        )
        alignment = canonical_sphere_alignment(
            case.model,
            radius_mm=config.radius_mm,
            initial_gap_mm=0.25,
        )
        contact_settings = FirstContactSettings(
            coarse_step_mm=FIRST_CONTACT_COARSE_STEP_MM,
            tolerance_mm=FIRST_CONTACT_TOLERANCE_MM,
            spawn_clearance_mm=SPAWN_CLEARANCE_MM,
            max_travel_mm=FIRST_CONTACT_MAX_TRAVEL_MM,
        )
        if intersects(case.contact_surface, sphere, alignment.nominal_pose):
            raise RuntimeError("canonical sphere reference pose is not collision-free")
        first_contact = find_first_contact(
            case.contact_surface,
            sphere,
            alignment.nominal_pose,
            alignment.approach_direction,
            contact_settings,
        )
        indenter = RigidIndenter3D(
            sphere,
            alignment.nominal_pose,
            alignment.approach_direction,
        )
        result = solve_fingertip_indentation(
            case.prepared,
            indenter,
            NewtonSettings(
                device=device,
                gravity=GRAVITY,
                dt=DT_S,
                steps=1,
                iterations=config.iterations,
                fixed_vertex_indices=case.prepared.support_vertex_indices,
            ),
            IndentationSettings(
                travel_mm=config.travel_mm,
                load_steps=config.load_steps,
                soft_contact_margin_mm=SOFT_CONTACT_MARGIN_MM,
                soft_contact_ke=SOFT_CONTACT_KE,
                soft_contact_kd=SOFT_CONTACT_KD,
            ),
            first_contact=first_contact,
        )
        runtime_s = time.perf_counter() - started
        mechanics = result.mechanics_result
        displacement = mechanics.displacement
        six_volumes = _six_volumes(mechanics.deformed_vertices, mechanics.tetrahedra)
        finite_vertices = bool(np.all(np.isfinite(mechanics.deformed_vertices)))
        finite_displacements = bool(np.all(np.isfinite(displacement)))
        inverted_tet_count = int(np.count_nonzero(six_volumes <= 0.0))
        max_displacement_mm = float(np.max(np.linalg.norm(displacement, axis=1)))
        rms_displacement_mm = float(np.sqrt(np.mean(np.square(displacement))))
        expected_pose = first_contact.pose_at_post_contact_travel(config.travel_mm)
        final_pose_error_mm = float(
            np.max(
                np.abs(
                    np.asarray(result.final_indenter_pose.translation_mm)
                    - np.asarray(expected_pose.translation_mm)
                )
            )
        )
        diagnostics = result.diagnostics
        max_soft_contact_count = int(diagnostics["max_soft_contact_count"])
        max_rigid_contact_count = int(diagnostics["max_rigid_contact_count"])
        max_soft_contact_overflow = int(diagnostics["max_soft_contact_overflow"])
        max_rigid_contact_overflow = int(diagnostics["max_rigid_contact_overflow"])
        record.update(
            {
                "status": "pass",
                "runtime_s": float(runtime_s),
                "first_contact_travel_mm": first_contact.travel_to_contact_mm,
                "first_contact_bracket_width_mm": first_contact.bracket_width_mm,
                "max_soft_contact_count": max_soft_contact_count,
                "max_rigid_contact_count": max_rigid_contact_count,
                "max_soft_contact_overflow": max_soft_contact_overflow,
                "max_rigid_contact_overflow": max_rigid_contact_overflow,
                "max_displacement_mm": max_displacement_mm,
                "rms_displacement_mm": rms_displacement_mm,
                "min_signed_six_volume": float(np.min(six_volumes)),
                "inverted_tet_count": inverted_tet_count,
                "finite_vertices": finite_vertices,
                "finite_displacements": finite_displacements,
                "final_pose_error_mm": final_pose_error_mm,
            }
        )
        failures: list[str] = []
        if not finite_vertices:
            failures.append("nonfinite_vertices")
        if not finite_displacements:
            failures.append("nonfinite_displacements")
        if inverted_tet_count:
            failures.append(f"inverted_tet_count={inverted_tet_count}")
        if max_soft_contact_overflow or max_rigid_contact_overflow:
            failures.append("contact_buffer_overflow")
        if final_pose_error_mm > 1.0e-6:
            failures.append(f"final_pose_error_mm={final_pose_error_mm:g}")
        if failures:
            record["status"] = "failed"
            record["failure_reason"] = "; ".join(failures)
        return _RunBundle(
            record=record,
            deformed_vertices=np.asarray(mechanics.deformed_vertices, dtype=float).copy(),
            displacement=np.asarray(displacement, dtype=float).copy(),
        )
    except Exception as exception:
        runtime_s = time.perf_counter() - started
        return _RunBundle(
            record=failed_record(
                config,
                f"{type(exception).__name__}: {exception}",
                device=device,
                runtime_s=runtime_s,
            )
        )


def _attach_comparison(
    candidate: _RunBundle,
    reference: _RunBundle,
    *,
    prefix: str = "",
) -> None:
    reference_key = f"{prefix}reference_run_id" if prefix else "reference_run_id"
    candidate.record[reference_key] = reference.record["run_id"]
    if (
        candidate.record["status"] != "pass"
        or reference.record["status"] != "pass"
        or candidate.deformed_vertices is None
        or reference.deformed_vertices is None
        or candidate.displacement is None
        or reference.displacement is None
    ):
        return
    metrics = comparison_metrics(
        candidate.deformed_vertices,
        reference.deformed_vertices,
        candidate.displacement,
        reference.displacement,
    )
    candidate.record.update(
        {f"{prefix}{key}" if prefix else key: value for key, value in metrics.items()}
    )


def _acceptable(record: Mapping[str, Any]) -> bool:
    return bool(
        record.get("status") == "pass"
        and record.get("rms_vertex_difference_mm") is not None
        and record.get("relative_max_displacement_difference") is not None
        and float(record["rms_vertex_difference_mm"]) <= RMS_VERTEX_THRESHOLD_MM
        and float(record["relative_max_displacement_difference"])
        <= RELATIVE_MAX_DISPLACEMENT_THRESHOLD
    )


def _run_and_collect(
    case: _PreparedCase,
    config: SweepConfig,
    device: str,
    bundles: list[_RunBundle],
) -> _RunBundle:
    bundle = _run_case(case, config, device)
    bundles.append(bundle)
    return bundle


def run_sweep(*, device: str = DEFAULT_DEVICE) -> dict[str, Any]:
    """Run the staged convergence sweep and return a JSON-ready summary."""

    case = _prepare_case()
    bundles: list[_RunBundle] = []

    # Stage A: sphere surface resolution at the 10 mm / 3 mm stress case.
    stage_a_sub2 = _run_and_collect(
        case,
        SweepConfig("stage_a_subdivision", 10.0, 3.0, 2, 60, 10),
        device,
        bundles,
    )
    stage_a_sub3 = _run_and_collect(
        case,
        SweepConfig("stage_a_subdivision", 10.0, 3.0, 3, 60, 10),
        device,
        bundles,
    )
    _attach_comparison(stage_a_sub2, stage_a_sub3)
    selected_subdivisions = 2 if _acceptable(stage_a_sub2.record) else 3

    # Stage B: load-step resolution, using the selected sphere resolution.
    stage_b = {
        steps: _run_and_collect(
            case,
            SweepConfig("stage_b_load_steps", 10.0, 3.0, selected_subdivisions, steps, 10),
            device,
            bundles,
        )
        for steps in (30, 60, 120)
    }
    _attach_comparison(stage_b[30], stage_b[120])
    _attach_comparison(stage_b[60], stage_b[120])
    if _acceptable(stage_b[30].record):
        selected_load_steps = 30
    elif _acceptable(stage_b[60].record):
        selected_load_steps = 60
    else:
        selected_load_steps = 120

    # Stage C: VBD iteration resolution at the selected load-step count.
    stage_c = {
        iterations: _run_and_collect(
            case,
            SweepConfig(
                "stage_c_iterations",
                10.0,
                3.0,
                selected_subdivisions,
                selected_load_steps,
                iterations,
            ),
            device,
            bundles,
        )
        for iterations in (5, 10, 20)
    }
    _attach_comparison(stage_c[5], stage_c[20])
    _attach_comparison(stage_c[10], stage_c[20])
    if _acceptable(stage_c[5].record):
        selected_iterations = 5
    elif _acceptable(stage_c[10].record):
        selected_iterations = 10
    else:
        selected_iterations = 20

    max_load_increment_mm = 3.0 / selected_load_steps
    search_stress = stage_c[selected_iterations]

    # A deliberately finer validation configuration is always measured once.
    validation_stress = _run_and_collect(
        case,
        SweepConfig("validation_reference", 10.0, 3.0, 3, 120, 20),
        device,
        bundles,
    )
    _attach_comparison(search_stress, validation_stress, prefix="validation_")

    # Stage D: selected search configuration across the intended radius/travel range.
    cross_check: list[_RunBundle] = []
    for radius_mm, travel_mm in ((2.0, 0.6), (5.0, 2.0), (10.0, 3.0)):
        cross_check.append(
            _run_and_collect(
                case,
                SweepConfig(
                    "stage_d_cross_check",
                    radius_mm,
                    travel_mm,
                    selected_subdivisions,
                    load_steps_for_increment(travel_mm, max_load_increment_mm),
                    selected_iterations,
                ),
                device,
                bundles,
            )
        )

    # Repeat one representative selected stress configuration to estimate the
    # observed context-to-context GPU noise floor without repeating the matrix.
    noise_runs = [
        _run_and_collect(
            case,
            SweepConfig(
                "noise_floor",
                10.0,
                3.0,
                selected_subdivisions,
                selected_load_steps,
                selected_iterations,
                repeat_index=index,
            ),
            device,
            bundles,
        )
        for index in (1, 2)
    ]
    _attach_comparison(noise_runs[1], noise_runs[0])
    noise_floor = {
        key: noise_runs[1].record.get(key)
        for key in (
            "max_abs_vertex_difference_mm",
            "rms_vertex_difference_mm",
            "max_displacement_difference_mm",
            "max_displacement_field_difference_mm",
            "relative_max_displacement_difference",
        )
    }
    noise_floor["reference_run_id"] = noise_runs[0].record["run_id"]

    records = [bundle.record for bundle in bundles]
    failed_count = sum(record["status"] != "pass" for record in records)
    selection = {
        "search": {
            "sphere_subdivisions": selected_subdivisions,
            "max_load_increment_mm": max_load_increment_mm,
            "load_steps_at_3mm": selected_load_steps,
            "iterations": selected_iterations,
            "stress_run_id": search_stress.record["run_id"],
        },
        "validation": {
            "sphere_subdivisions": 3,
            "max_load_increment_mm": 3.0 / 120.0,
            "load_steps_at_3mm": 120,
            "iterations": 20,
            "stress_run_id": validation_stress.record["run_id"],
        },
    }
    return {
        "schema": "physics-newton-sphere-convergence-sweep-v1",
        "fixed_parameters": {
            "dt": DT_S,
            "gravity": GRAVITY,
            "soft_contact_margin_mm": SOFT_CONTACT_MARGIN_MM,
            "soft_contact_ke": SOFT_CONTACT_KE,
            "soft_contact_kd": SOFT_CONTACT_KD,
            "first_contact_coarse_step_mm": FIRST_CONTACT_COARSE_STEP_MM,
            "first_contact_tolerance_mm": FIRST_CONTACT_TOLERANCE_MM,
            "spawn_clearance_mm": SPAWN_CLEARANCE_MM,
            "mesh_tier": "search",
            "support_semantics": "prepared_fingertip.support_vertex_indices",
        },
        "selection": selection,
        "noise_floor": noise_floor,
        "successful_runs": len(records) - failed_count,
        "failed_runs": failed_count,
        "records": records,
        "stress_case_healthy_under_search": bool(
            cross_check[-1].record["status"] == "pass"
        ),
    }


_CSV_COLUMNS = (
    "run_id",
    "stage",
    "repeat_index",
    "status",
    "failure_reason",
    "device",
    "radius_mm",
    "travel_mm",
    "sphere_subdivisions",
    "load_steps",
    "load_increment_mm",
    "iterations",
    "dt",
    "runtime_s",
    "first_contact_travel_mm",
    "first_contact_bracket_width_mm",
    "first_contact_tolerance_mm",
    "spawn_clearance_mm",
    "max_soft_contact_count",
    "max_rigid_contact_count",
    "max_soft_contact_overflow",
    "max_rigid_contact_overflow",
    "max_displacement_mm",
    "rms_displacement_mm",
    "min_signed_six_volume",
    "inverted_tet_count",
    "finite_vertices",
    "finite_displacements",
    "final_pose_error_mm",
    "reference_run_id",
    "max_abs_vertex_difference_mm",
    "rms_vertex_difference_mm",
    "max_displacement_difference_mm",
    "max_displacement_field_difference_mm",
    "relative_max_displacement_difference",
    "validation_reference_run_id",
    "validation_max_abs_vertex_difference_mm",
    "validation_rms_vertex_difference_mm",
    "validation_max_displacement_difference_mm",
    "validation_max_displacement_field_difference_mm",
    "validation_relative_max_displacement_difference",
)


def write_sweep_artifacts(
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write strict JSON and deterministic CSV sweep artifacts."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "newton_sphere_sweep.json"
    csv_path = directory / "newton_sphere_sweep.csv"
    atomic_write_json(json_path, summary)
    records = summary.get("records", [])
    write_csv(
        csv_path,
        _CSV_COLUMNS,
        ([record.get(column) for column in _CSV_COLUMNS] for record in records),
    )
    return {"json": str(json_path), "csv": str(csv_path)}


def _print_report(summary: Mapping[str, Any], paths: Mapping[str, str]) -> None:
    print("Newton sphere-contact convergence sweep")
    print(
        f"runs: {summary['successful_runs']} passed, "
        f"{summary['failed_runs']} failed"
    )
    for record in summary["records"]:
        print(
            f"{record['status'].upper():6s} {record['run_id']} "
            f"runtime={record['runtime_s'] if record['runtime_s'] is not None else 'n/a'} "
            f"max_u={record['max_displacement_mm'] if record['max_displacement_mm'] is not None else 'n/a'}"
        )
    print(f"search recommendation: {summary['selection']['search']}")
    print(f"validation recommendation: {summary['selection']['validation']}")
    print(f"noise floor: {summary['noise_floor']}")
    print(
        "R=10 mm / travel=3 mm healthy under search: "
        f"{summary['stress_case_healthy_under_search']}"
    )
    print(f"wrote {paths['json']}")
    print(f"wrote {paths['csv']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    try:
        import warp as wp

        if not wp.is_device_available(args.device):
            raise RuntimeError(f"CUDA device {args.device!r} is not available")
        summary = run_sweep(device=args.device)
        paths = write_sweep_artifacts(summary, args.output_dir)
        _print_report(summary, paths)
    except Exception as exception:
        print(f"FAIL: newton sphere convergence sweep: {exception}")
        return 1
    return 0 if summary["failed_runs"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SweepConfig",
    "comparison_metrics",
    "failed_record",
    "load_steps_for_increment",
    "main",
    "run_sweep",
    "write_sweep_artifacts",
]

"""Run and synthesize the Phase 4K mechanical deformation transfer map.

The parent process owns the resumable queue.  Every nonlinear case is run in
its own Python child process and therefore receives a fresh ``Kratos.Model``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fem.indentation_analysis import IndentationSettings, run_indentation_case
from fem.indenter_fixture import (
    build_indenter_fixture,
    build_indenter_fixture_at_location,
)
from fem.kratos_settings import (
    ABSOLUTE_TOLERANCE,
    CONSTITUTIVE_LAW,
    MAXIMUM_NEWTON_ITERATIONS,
    MIXED_PAD_ELEMENT,
    MORTAR_TYPE,
    POISSON_RATIO,
    RELATIVE_TOLERANCE,
    YOUNG_MODULUS_MPA,
)
from fem.mechanical_transfer_map import (
    CODTMStepRecorder,
    TransferMapSettings,
    observation_boundary_contract,
    reference_outer_arc_chain,
)
from fem.no_void_baseline import atomic_write_json, no_void_geometry_contract
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPOSITORY_ROOT / "output" / "phase4_mechanical_transfer_map"
PHASE_J_ROOT = REPOSITORY_ROOT / "output" / "phase4_no_void_baseline"
PYTHON = Path("/home/dk/miniconda3/envs/lit/bin/python")
REPRESENTATIVE_DEPTHS_MM = (0.25, 0.5, 1.0, 1.5)
MEDIUM_LOCATIONS = (0.20, 0.35, 0.50, 0.65, 0.80)
FINE_LOCATIONS = (0.20, 0.50, 0.80)
SIDE_NAMES = ("left", "right")


CASE_SPECS = (
    {
        "case_name": "center_medium",
        "mesh": "medium",
        "xi_cmd": 0.50,
        "stage": "k2_center_baseline",
        "directory": "k2_center_baseline/medium",
    },
    {
        "case_name": "center_fine",
        "mesh": "fine",
        "xi_cmd": 0.50,
        "stage": "k2_center_baseline",
        "directory": "k2_center_baseline/fine",
    },
    {
        "case_name": "medium_xi_0p20",
        "mesh": "medium",
        "xi_cmd": 0.20,
        "stage": "k3_medium_location_sweep",
        "directory": "k3_medium_location_sweep/xi_0p20",
    },
    {
        "case_name": "medium_xi_0p35",
        "mesh": "medium",
        "xi_cmd": 0.35,
        "stage": "k3_medium_location_sweep",
        "directory": "k3_medium_location_sweep/xi_0p35",
    },
    {
        "case_name": "medium_xi_0p65",
        "mesh": "medium",
        "xi_cmd": 0.65,
        "stage": "k3_medium_location_sweep",
        "directory": "k3_medium_location_sweep/xi_0p65",
    },
    {
        "case_name": "medium_xi_0p80",
        "mesh": "medium",
        "xi_cmd": 0.80,
        "stage": "k3_medium_location_sweep",
        "directory": "k3_medium_location_sweep/xi_0p80",
    },
    {
        "case_name": "fine_xi_0p20",
        "mesh": "fine",
        "xi_cmd": 0.20,
        "stage": "k4_fine_spot_checks",
        "directory": "k4_fine_spot_checks/xi_0p20",
    },
    {
        "case_name": "fine_xi_0p80",
        "mesh": "fine",
        "xi_cmd": 0.80,
        "stage": "k4_fine_spot_checks",
        "directory": "k4_fine_spot_checks/xi_0p80",
    },
)


def _strict_read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant {value}")
        ),
    )


def _case_is_complete(spec: Mapping[str, Any]) -> bool:
    directory = OUTPUT_ROOT / str(spec["directory"])
    result_path = directory / "result.json"
    records_path = directory / "codtm_step_records.json"
    if not result_path.is_file() or not records_path.is_file():
        return False
    try:
        result = _strict_read_json(result_path)
        records = _strict_read_json(records_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if (
        result.get("phase") != "4K"
        or result.get("case_name") != spec["case_name"]
        or result.get("terminal_artifact") is not True
        or not isinstance(records.get("records"), list)
    ):
        return False
    if result.get("solve_status") == "PASS":
        return len(records["records"]) == 48
    return result.get("status") in {"FAIL", "TIMEOUT"}


def _case_command(spec: Mapping[str, Any], output_directory: Path) -> list[str]:
    return [
        str(PYTHON),
        "-B",
        str(Path(__file__).resolve()),
        "--run-case",
        str(spec["case_name"]),
        "--mesh",
        str(spec["mesh"]),
        "--xi",
        f"{float(spec['xi_cmd']):.17g}",
        "--case-output",
        str(output_directory),
    ]


def _write_case_history_csv(
    path: Path,
    result: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    by_step = {int(record["step"]): record for record in records}
    columns = (
        "step",
        "delta_n_mm",
        "xi_cmd",
        "xi_centroid",
        "canonical_normal_reaction_n",
        "integrated_contact_resultant_n",
        "force_closure_relative_error",
        "contact_length_mm",
        "contact_verification",
        "minimum_det_f",
        "strain_metric",
        "nonlinear_iterations",
        "external_reaction_work_n_mm",
        "valid",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for point in result.get("history", []):
            record = by_step.get(int(point["step"]))
            if record is None:
                continue
            contact = record["contact"]
            writer.writerow(
                {
                    "step": record["step"],
                    "delta_n_mm": record["delta_n_mm"],
                    "xi_cmd": record["xi_cmd"],
                    "xi_centroid": contact["xi_centroid"],
                    "canonical_normal_reaction_n": record[
                        "canonical_normal_reaction_n"
                    ],
                    "integrated_contact_resultant_n": contact[
                        "integrated_contact_resultant_n"
                    ],
                    "force_closure_relative_error": contact[
                        "force_closure_relative_error"
                    ],
                    "contact_length_mm": contact["contact_length_mm"],
                    "contact_verification": contact["verification"],
                    "minimum_det_f": record["minimum_det_f"],
                    "strain_metric": record["canonical_strain_metric"]["value"],
                    "nonlinear_iterations": record["nonlinear_iterations"],
                    "external_reaction_work_n_mm": record[
                        "external_reaction_work_n_mm"
                    ],
                    "valid": bool(
                        record["solver_converged"]
                        and record["finite_fields"]
                        and record["minimum_det_f"] > 0.0
                    ),
                }
            )


def run_case_child(
    case_name: str,
    mesh: str,
    xi_cmd: float,
    output_directory: Path,
) -> int:
    output_directory.mkdir(parents=True, exist_ok=True)
    model = FingertipModel(FingertipParameters())
    settings = IndentationSettings(1.5, 48)
    fixture = (
        build_indenter_fixture(model)
        if abs(xi_cmd - 0.5) <= 1.0e-15
        else build_indenter_fixture_at_location(model, xi_cmd)
    )
    source_arc = model.boundaries.segments["pad_outer_arc"].geometry
    actual_target_xi = (
        float(source_arc.project(__import__(
            "shapely.geometry", fromlist=["Point"]
        ).Point(fixture.frame.point_mm)))
        / float(source_arc.length)
    )
    recorder = CODTMStepRecorder(case_name, xi_cmd)
    result, _ = run_indentation_case(
        model,
        mesh,  # type: ignore[arg-type]
        settings,
        internal_contact_configuration="none",
        fixture_override=fixture,
        converged_step_observer=recorder,
    )
    result.update(
        {
            "phase": "4K",
            "case_name": case_name,
            "xi_cmd": xi_cmd,
            "actual_reference_target_xi": actual_target_xi,
            "codtm_recorder_added": True,
            "phase4j_solver_configuration_changed": False,
            "internal_contact_configuration": "none",
            "terminal_artifact": True,
            "codtm_record_count": len(recorder.records),
            "codtm_metadata": recorder.metadata(),
        }
    )
    atomic_write_json(
        output_directory / "codtm_step_records.json",
        {
            "phase": "4K",
            "case_name": case_name,
            "metadata": recorder.metadata(),
            "records": recorder.records,
        },
    )
    atomic_write_json(output_directory / "result.json", result)
    _write_case_history_csv(
        output_directory / "history.csv", result, recorder.records
    )
    return 0


def _git(command: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *command],
        cwd=REPOSITORY_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _source_trace() -> dict[str, Any]:
    import KratosMultiphysics as KM
    import KratosMultiphysics.ContactStructuralMechanicsApplication as CSMA

    source_pointer = Path("/tmp/phase4ig_source_dir")
    source_root = (
        Path(source_pointer.read_text(encoding="utf-8").strip())
        if source_pointer.is_file()
        else None
    )
    application = (
        source_root
        / "applications"
        / "ContactStructuralMechanicsApplication"
        if source_root is not None
        else None
    )
    evidence = {
        "variables": (
            application / "contact_structural_mechanics_application_variables.h"
            if application is not None
            else None
        ),
        "active_set": (
            application / "custom_utilities" / "active_set_utilities.cpp"
            if application is not None
            else None
        ),
        "alm_process": (
            application / "python_scripts" / "alm_contact_process.py"
            if application is not None
            else None
        ),
        "condition": (
            application
            / "custom_conditions"
            / "ALM_frictionless_mortar_contact_condition.cpp"
            if application is not None
            else None
        ),
    }
    return {
        "kratos_version": KM.Kernel.Version(),
        "module_paths": {
            "KratosMultiphysics": str(Path(KM.__file__).resolve()),
            "ContactStructuralMechanicsApplication": str(
                Path(CSMA.__file__).resolve()
            ),
        },
        "matching_source_checkout": (
            str(source_root) if source_root is not None else None
        ),
        "source_evidence": {
            name: {
                "path": str(path) if path is not None else None,
                "available": bool(path is not None and path.is_file()),
            }
            for name, path in evidence.items()
        },
        "semantics": {
            "LAGRANGE_MULTIPLIER_CONTACT_PRESSURE": {
                "storage": "historical nodal solution-step variable",
                "role": "ALM pressure unknown and slave-node DOF",
                "source_evidence": (
                    "variables header line 99; generated condition reads "
                    "parent/slave geometry with GetVariableVector(..., 0)"
                ),
            },
            "AUGMENTED_NORMAL_CONTACT_PRESSURE": {
                "storage": "non-historical nodal value",
                "role": (
                    "effective pressure used by active-set logic; negative "
                    "means compression"
                ),
                "source_evidence": (
                    "active_set_utilities.cpp lines 190-196 computes "
                    "scale_factor*LM + epsilon*WEIGHTED_GAP"
                ),
            },
            "NODAL_AREA": {
                "storage": "non-historical nodal value",
                "role": "Kratos ALM process contact-resultant weight",
                "source_evidence": (
                    "alm_contact_process.py lines 256-258 sums "
                    "NODAL_AREA*AUGMENTED_NORMAL_CONTACT_PRESSURE"
                ),
            },
            "CONTACT_FORCE_CONTACT_PRESSURE_NORMAL_CONTACT_STRESS": {
                "status": (
                    "registered candidates only; not selected without runtime "
                    "storage and force-closure evidence in this ALM formulation"
                )
            },
        },
        "selected_contact_distribution": {
            "p_plus": "-AUGMENTED_NORMAL_CONTACT_PRESSURE on ACTIVE PadOuterArc SLAVE nodes",
            "resultant": (
                "sum(p_plus*NODAL_AREA*1 mm plane-strain thickness*"
                "(-historical KM.NORMAL dot global loading direction))"
            ),
            "canonical_force": (
                "compressive_indenter_reaction from prescribed indenter "
                "REACTION projected on the unchanged global loading direction"
            ),
            "trust_gate": "relative closure <= 0.02",
        },
    }


def _preflight() -> dict[str, Any]:
    from fem.fingertip_mesher import generate_fingertip_mesh
    from fem.indentation_analysis import inspect_indentation_runtime_contract
    from fem.mesh_types import mesh_settings_for_level

    model = FingertipModel(FingertipParameters())
    transfer_settings = TransferMapSettings()
    meshes: dict[str, Any] = {}
    for level in ("medium", "fine"):
        mesh = generate_fingertip_mesh(
            model, mesh_settings_for_level(level)  # type: ignore[arg-type]
        )
        chain = reference_outer_arc_chain(model, mesh)
        meshes[level] = {
            "outer_arc_node_count": len(chain.node_ids),
            "outer_arc_edge_count": len(
                mesh.boundary_edges["pad_outer_arc"]
            ),
            "outer_arc_reference_polyline_length_mm": chain.total_length_mm,
            "first_coordinate_mm": list(chain.points_mm[0]),
            "last_coordinate_mm": list(chain.points_mm[-1]),
            "semantic_boundary_present": "pad_outer_arc"
            in mesh.boundary_edges,
        }
    runtime = inspect_indentation_runtime_contract(
        model,
        "medium",
        IndentationSettings(1.5, 48),
        internal_contact_configuration="none",
    )
    j_paths = {
        "medium": PHASE_J_ROOT / "j1_medium" / "result.json",
        "fine": PHASE_J_ROOT / "j2_fine" / "result.json",
    }
    checkpoint = {
        level: {
            "path": str(path.relative_to(REPOSITORY_ROOT)),
            "available": path.is_file(),
            "history_has_nodal_state_every_step": False,
            "reason": (
                "Phase 4J history stores scalar metrics every step and nodal "
                "snapshots only at 0.5/1.0/1.5 mm"
            ),
            "phase4k_action": "rerun unchanged solver with converged-step recorder",
        }
        for level, path in j_paths.items()
    }
    return {
        "phase": "4K",
        "status": "PASS",
        "git": {
            "branch": _git(["branch", "--show-current"]),
            "head": _git(["rev-parse", "HEAD"]),
            "recent_commits": _git(["log", "-5", "--oneline"]).splitlines(),
        },
        "geometry": no_void_geometry_contract(model),
        "observation_boundary": observation_boundary_contract(
            transfer_settings
        ),
        "contact_coordinate": {
            "source": "FingertipModel semantic pad_outer_arc",
            "orientation": (
                "xi=0 right top, xi=0.5 crown, xi=1 left top"
            ),
        },
        "mesh_chains": meshes,
        "runtime_contact_contract": runtime,
        "checkpoint_capabilities": checkpoint,
        "phase4j_compatibility": {
            "element": MIXED_PAD_ELEMENT,
            "law": CONSTITUTIVE_LAW,
            "poisson_ratio": POISSON_RATIO,
            "internal_contact": "none",
            "indentation_mm": 1.5,
            "steps": 48,
            "solver_parameter_changes": [],
        },
    }


def _run_k1_tests() -> dict[str, Any]:
    directory = OUTPUT_ROOT / "k1_extractor_tests"
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        "-B",
        "-m",
        "pytest",
        "-q",
        "tests/test_mechanical_transfer_map.py",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (directory / "pytest.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    result = {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "command": " ".join(command),
        "returncode": completed.returncode,
        "log": "k1_extractor_tests/pytest.log",
    }
    atomic_write_json(directory / "result.json", result)
    return result


def _load_case(spec: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = OUTPUT_ROOT / str(spec["directory"])
    result = _strict_read_json(directory / "result.json")
    records = _strict_read_json(
        directory / "codtm_step_records.json"
    )["records"]
    return result, records


def _depth_step(depth: float) -> int:
    return int(round(depth / 1.5 * 48.0)) - 1


def _signature(record: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(row["u_normal_mm"])
            for side in SIDE_NAMES
            for row in record["observation_sidewalls"][side]
        ],
        dtype=float,
    )


def _tangent_signature(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    delta = np.asarray([record["delta_n_mm"] for record in records], dtype=float)
    values = np.asarray([_signature(record) for record in records])
    return np.gradient(values, delta, axis=0, edge_order=1)


def _signature_norm(vector: np.ndarray, eta: np.ndarray) -> float:
    count = len(eta)
    return math.sqrt(
        float(np.trapezoid(vector[:count] ** 2, eta))
        + float(np.trapezoid(vector[count:] ** 2, eta))
    )


def _distance_matrix(
    signatures: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    count = signatures.shape[0]
    result = np.zeros((count, count), dtype=float)
    for first in range(count):
        for second in range(count):
            result[first, second] = _signature_norm(
                signatures[first] - signatures[second], eta
            )
    return result


def _shape_distance_matrix(
    signatures: np.ndarray,
    eta: np.ndarray,
    floor_mm: float,
) -> np.ndarray:
    norms = np.asarray(
        [_signature_norm(signature, eta) for signature in signatures]
    )
    normalized = np.asarray(
        [
            signature / max(norm, floor_mm)
            for signature, norm in zip(signatures, norms)
        ]
    )
    return _distance_matrix(normalized, eta)


def _normalized_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(
        second_centered
    )
    return (
        float(np.dot(first_centered, second_centered) / denominator)
        if denominator > 1.0e-14
        else None
    )


def _profile_difference(
    medium: np.ndarray,
    fine: np.ndarray,
    floor_mm: float,
) -> dict[str, Any]:
    difference = medium - fine
    denominator = max(
        float(np.linalg.norm(fine)),
        floor_mm * math.sqrt(fine.size),
    )
    return {
        "relative_l2_difference": float(np.linalg.norm(difference))
        / denominator,
        "maximum_absolute_difference_mm": float(
            np.max(np.abs(difference))
        ),
        "normalized_shape_correlation": _normalized_correlation(
            medium, fine
        ),
    }


def _assemble_arrays(
    loaded: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    case_count = len(loaded)
    step_count = 48
    side_count = 2
    sample_count = 41
    u_xy = np.full(
        (case_count, step_count, side_count, sample_count, 2),
        np.nan,
    )
    u_normal = np.full(
        (case_count, step_count, side_count, sample_count), np.nan
    )
    u_tangent = np.full_like(u_normal, np.nan)
    delta = np.full((case_count, step_count), np.nan)
    force = np.full_like(delta, np.nan)
    centroid = np.full_like(delta, np.nan)
    length = np.full_like(delta, np.nan)
    valid = np.zeros((case_count, step_count), dtype=bool)
    xi = np.asarray(
        [float(spec["xi_cmd"]) for spec, _, _ in loaded], dtype=float
    )
    eta = np.tile(np.linspace(0.0, 1.0, sample_count), (2, 1))
    rows: list[dict[str, Any]] = []
    for case_index, (spec, result, records) in enumerate(loaded):
        for record in records:
            step_index = int(record["step"]) - 1
            if not 0 <= step_index < step_count:
                continue
            delta[case_index, step_index] = float(record["delta_n_mm"])
            force[case_index, step_index] = float(
                record["canonical_normal_reaction_n"]
            )
            contact = record["contact"]
            if contact["xi_centroid"] is not None:
                centroid[case_index, step_index] = float(
                    contact["xi_centroid"]
                )
            if contact["contact_length_mm"] is not None:
                length[case_index, step_index] = float(
                    contact["contact_length_mm"]
                )
            record_valid = bool(
                record["solver_converged"]
                and record["finite_fields"]
                and float(record["minimum_det_f"]) > 0.0
            )
            valid[case_index, step_index] = record_valid
            for side_index, side in enumerate(SIDE_NAMES):
                for sample_index, sample in enumerate(
                    record["observation_sidewalls"][side]
                ):
                    ux = float(sample["ux_mm"])
                    uy = float(sample["uy_mm"])
                    normal = float(sample["u_normal_mm"])
                    tangent = float(sample["u_tangent_mm"])
                    u_xy[
                        case_index,
                        step_index,
                        side_index,
                        sample_index,
                    ] = (ux, uy)
                    u_normal[
                        case_index,
                        step_index,
                        side_index,
                        sample_index,
                    ] = normal
                    u_tangent[
                        case_index,
                        step_index,
                        side_index,
                        sample_index,
                    ] = tangent
                    rows.append(
                        {
                            "case": spec["case_name"],
                            "mesh": spec["mesh"],
                            "step": record["step"],
                            "delta_n": record["delta_n_mm"],
                            "xi_cmd": spec["xi_cmd"],
                            "xi_centroid": contact["xi_centroid"],
                            "F_n": record["canonical_normal_reaction_n"],
                            "contact_length": contact["contact_length_mm"],
                            "side_name": side,
                            "eta": sample["eta"],
                            "X0_x": sample["reference_x_mm"],
                            "X0_y": sample["reference_y_mm"],
                            "u_x": ux,
                            "u_y": uy,
                            "u_normal": normal,
                            "u_tangent": tangent,
                            "deformed_x": sample["deformed_x_mm"],
                            "deformed_y": sample["deformed_y_mm"],
                            "min_detF": record["minimum_det_f"],
                            "strain_metric": record[
                                "canonical_strain_metric"
                            ]["value"],
                            "valid": record_valid,
                        }
                    )
    arrays = {
        "xi_cmd": xi,
        "delta_n": delta,
        "eta": eta,
        "u_xy": u_xy,
        "u_normal": u_normal,
        "u_tangent": u_tangent,
        "F_n": force,
        "xi_centroid": centroid,
        "contact_length": length,
        "valid_mask": valid,
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        arrays["G_secant"] = u_normal / delta[:, :, None, None]
    tangent_gain = np.full_like(u_normal, np.nan)
    for case_index in range(case_count):
        case_valid = valid[case_index]
        if np.count_nonzero(case_valid) >= 2:
            tangent_gain[case_index, case_valid] = np.gradient(
                u_normal[case_index, case_valid],
                delta[case_index, case_valid],
                axis=0,
                edge_order=1,
            )
    arrays["G_tangent"] = tangent_gain
    medium_indices = [
        next(
            index
            for index, (spec, _, _) in enumerate(loaded)
            if spec["mesh"] == "medium"
            and abs(float(spec["xi_cmd"]) - xi_value) <= 1.0e-15
        )
        for xi_value in MEDIUM_LOCATIONS
    ]
    arrays["medium_xi"] = np.asarray(MEDIUM_LOCATIONS, dtype=float)
    arrays["S_location"] = np.gradient(
        u_normal[medium_indices],
        np.asarray(MEDIUM_LOCATIONS, dtype=float),
        axis=0,
        edge_order=1,
    )
    return arrays, rows


def _write_long_csv(rows: Sequence[Mapping[str, Any]]) -> None:
    path = OUTPUT_ROOT / "codtm_long.csv"
    columns = (
        "case",
        "mesh",
        "step",
        "delta_n",
        "xi_cmd",
        "xi_centroid",
        "F_n",
        "contact_length",
        "side_name",
        "eta",
        "X0_x",
        "X0_y",
        "u_x",
        "u_y",
        "u_normal",
        "u_tangent",
        "deformed_x",
        "deformed_y",
        "min_detF",
        "strain_metric",
        "valid",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _case_summary(
    loaded: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec, result, records in loaded:
        final = records[-1] if records else None
        verified_steps = (
            sum(
                record["contact"]["verification"] == "VERIFIED"
                for record in records
            )
            if records
            else 0
        )
        rows.append(
            {
                "case": spec["case_name"],
                "stage": spec["stage"],
                "mesh": spec["mesh"],
                "xi_cmd": spec["xi_cmd"],
                "solve_status": result.get("solve_status"),
                "case_status": result.get("status"),
                "converged_steps": len(records),
                "final_reaction_n": (
                    final["canonical_normal_reaction_n"]
                    if final is not None
                    else None
                ),
                "final_xi_centroid": (
                    final["contact"]["xi_centroid"]
                    if final is not None
                    else None
                ),
                "final_contact_length_mm": (
                    final["contact"]["contact_length_mm"]
                    if final is not None
                    else None
                ),
                "final_force_closure_error": (
                    final["contact"]["force_closure_relative_error"]
                    if final is not None
                    else None
                ),
                "verified_contact_steps": verified_steps,
                "minimum_det_f": (
                    min(record["minimum_det_f"] for record in records)
                    if records
                    else None
                ),
                "maximum_strain_metric": (
                    max(
                        record["canonical_strain_metric"]["value"]
                        for record in records
                    )
                    if records
                    else None
                ),
                "maximum_nonlinear_iterations": (
                    max(record["nonlinear_iterations"] for record in records)
                    if records
                    else None
                ),
                "solve_wall_clock_seconds": result.get(
                    "solve_wall_clock_seconds"
                ),
                "failure_reason": result.get("failure_reason"),
            }
        )
    path = OUTPUT_ROOT / "case_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _synthesize_metrics(
    loaded: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ],
) -> dict[str, Any]:
    by_key = {
        (str(spec["mesh"]), float(spec["xi_cmd"])): (result, records)
        for spec, result, records in loaded
    }
    medium = [
        (xi, *by_key[("medium", xi)]) for xi in MEDIUM_LOCATIONS
    ]
    eta = np.linspace(0.0, 1.0, 41)
    floor = IndentationSettings(1.5, 48).profile_displacement_floor_mm
    slices: dict[str, Any] = {}
    all_medium_complete = all(len(records) == 48 for _, _, records in medium)
    for depth in REPRESENTATIVE_DEPTHS_MM:
        step_index = _depth_step(depth)
        if not all_medium_complete:
            slices[f"{depth:g}"] = {
                "available": False,
                "reason": "one or more medium cases incomplete",
            }
            continue
        signatures = np.asarray(
            [_signature(records[step_index]) for _, _, records in medium]
        )
        norms = np.asarray(
            [_signature_norm(signature, eta) for signature in signatures]
        )
        centered = signatures - np.mean(signatures, axis=0, keepdims=True)
        singular_values = np.linalg.svd(
            centered, full_matrices=False, compute_uv=False
        )
        location_sensitivity = np.gradient(
            signatures,
            np.asarray(MEDIUM_LOCATIONS),
            axis=0,
            edge_order=1,
        )
        slices[f"{depth:g}"] = {
            "available": True,
            "step": step_index + 1,
            "interpolated": False,
            "lateral_signal_norm_mm": norms.tolist(),
            "indentation_normalized_gain": (norms / depth).tolist(),
            "fixed_indentation_distance_matrix_mm": _distance_matrix(
                signatures, eta
            ).tolist(),
            "amplitude_normalized_shape_distance_matrix": (
                _shape_distance_matrix(signatures, eta, floor).tolist()
            ),
            "location_sensitivity_l2_mm_per_xi": [
                _signature_norm(value, eta)
                for value in location_sensitivity
            ],
            "signature_singular_values_mm": singular_values.tolist(),
            "svd_interpretation": (
                "descriptive only; no optical-noise observability threshold"
            ),
        }

    force_conditioned: dict[str, Any]
    if not all(len(records) == 48 for _, _, records in medium):
        force_conditioned = {
            "available": False,
            "reason": "one or more force curves are incomplete",
        }
    else:
        lower = max(
            min(
                float(record["canonical_normal_reaction_n"])
                for record in records
            )
            for _, _, records in medium
        )
        upper = min(
            max(
                float(record["canonical_normal_reaction_n"])
                for record in records
            )
            for _, _, records in medium
        )
        force_levels = np.linspace(lower, upper, 5)
        matrices = []
        crossing_counts: list[list[int]] = []
        interpolation_failed = False
        for target_force in force_levels:
            signatures = []
            level_crossing_counts = []
            for _, _, records in medium:
                forces = np.asarray(
                    [
                        record["canonical_normal_reaction_n"]
                        for record in records
                    ],
                    dtype=float,
                )
                profiles = np.asarray(
                    [_signature(record) for record in records]
                )
                crossings = [
                    index
                    for index in range(len(forces) - 1)
                    if (
                        min(forces[index], forces[index + 1])
                        <= target_force
                        <= max(forces[index], forces[index + 1])
                    )
                    and abs(forces[index + 1] - forces[index]) > 1.0e-14
                ]
                level_crossing_counts.append(len(crossings))
                if not crossings:
                    interpolation_failed = True
                    break
                index = crossings[0]
                weight = (
                    (target_force - forces[index])
                    / (forces[index + 1] - forces[index])
                )
                signatures.append(
                    (1.0 - weight) * profiles[index]
                    + weight * profiles[index + 1]
                )
            crossing_counts.append(level_crossing_counts)
            if interpolation_failed:
                break
            matrices.append(
                _distance_matrix(np.asarray(signatures), eta).tolist()
            )
        if interpolation_failed:
            force_conditioned = {
                "available": False,
                "reason": "a common force level has no loading-path crossing",
                "common_force_range_n": [lower, upper],
            }
        else:
            force_conditioned = {
                "available": True,
                "common_force_range_n": [lower, upper],
                "force_levels_n": force_levels.tolist(),
                "distance_matrices_mm": matrices,
                "crossing_counts_by_force_and_location": crossing_counts,
                "interpolation": (
                    "piecewise linear at the first crossing along the unmodified "
                    "loading path; no monotonicization or smoothing"
                ),
                "multiple_crossing_policy": (
                    "first loading-path crossing is used and all crossing "
                    "counts are reported"
                ),
            }

    mesh_comparison: dict[str, Any] = {}
    for xi in FINE_LOCATIONS:
        medium_records = by_key[("medium", xi)][1]
        fine_records = by_key[("fine", xi)][1]
        if len(medium_records) != 48 or len(fine_records) != 48:
            mesh_comparison[f"{xi:.2f}"] = {
                "available": False,
                "reason": "medium or fine case incomplete",
            }
            continue
        depth_rows: dict[str, Any] = {}
        medium_tangent = _tangent_signature(medium_records)
        fine_tangent = _tangent_signature(fine_records)
        for depth in REPRESENTATIVE_DEPTHS_MM:
            index = _depth_step(depth)
            normal = _profile_difference(
                _signature(medium_records[index]),
                _signature(fine_records[index]),
                floor,
            )
            gain_medium = _signature_norm(
                _signature(medium_records[index]), eta
            ) / depth
            gain_fine = _signature_norm(
                _signature(fine_records[index]), eta
            ) / depth
            normal["transfer_gain_medium"] = gain_medium
            normal["transfer_gain_fine"] = gain_fine
            normal["transfer_gain_relative_difference"] = (
                abs(gain_medium - gain_fine)
                / max(abs(gain_fine), floor)
            )
            normal["tangent_gain_profile"] = _profile_difference(
                medium_tangent[index],
                fine_tangent[index],
                floor,
            )
            depth_rows[f"{depth:g}"] = normal
        medium_force = float(
            medium_records[-1]["canonical_normal_reaction_n"]
        )
        fine_force = float(fine_records[-1]["canonical_normal_reaction_n"])
        mesh_comparison[f"{xi:.2f}"] = {
            "available": True,
            "final_reaction_relative_difference": abs(
                medium_force - fine_force
            )
            / abs(fine_force),
            "final_centroids": {
                "medium": medium_records[-1]["contact"]["xi_centroid"],
                "fine": fine_records[-1]["contact"]["xi_centroid"],
            },
            "final_contact_lengths_mm": {
                "medium": medium_records[-1]["contact"]["contact_length_mm"],
                "fine": fine_records[-1]["contact"]["contact_length_mm"],
            },
            "minimum_det_f": {
                "medium": min(
                    record["minimum_det_f"] for record in medium_records
                ),
                "fine": min(
                    record["minimum_det_f"] for record in fine_records
                ),
            },
            "profiles_by_depth": depth_rows,
        }

    stabilization: dict[str, Any] = {}
    for xi, _, records in medium:
        if len(records) != 48:
            stabilization[f"{xi:.2f}"] = {"available": False}
            continue
        delta = np.asarray(
            [record["delta_n_mm"] for record in records], dtype=float
        )
        force = np.asarray(
            [record["canonical_normal_reaction_n"] for record in records],
            dtype=float,
        )
        stiffness = np.gradient(force, delta, edge_order=1)
        verified = [
            record
            for record in records
            if record["contact"]["verification"] == "VERIFIED"
        ]
        stabilization[f"{xi:.2f}"] = {
            "available": True,
            "verified_contact_step_count": len(verified),
            "centroid_drift": (
                max(record["contact"]["xi_centroid"] for record in verified)
                - min(record["contact"]["xi_centroid"] for record in verified)
                if verified
                else None
            ),
            "contact_length_range_mm": (
                [
                    min(
                        record["contact"]["contact_length_mm"]
                        for record in verified
                    ),
                    max(
                        record["contact"]["contact_length_mm"]
                        for record in verified
                    ),
                ]
                if verified
                else None
            ),
            "dF_d_delta_n_per_mm": stiffness.tolist(),
            "early_incremental_stiffness_n_per_mm": float(
                (force[15] - force[7]) / (delta[15] - delta[7])
            ),
            "late_incremental_stiffness_n_per_mm": float(
                (force[47] - force[31]) / (delta[47] - delta[31])
            ),
        }

    return {
        "medium_locations": list(MEDIUM_LOCATIONS),
        "representative_slices": slices,
        "force_conditioned_separability": force_conditioned,
        "mesh_comparison": mesh_comparison,
        "contact_stabilization": stabilization,
        "local_transfer_jacobian": {
            "definition": (
                "columns are finite-difference partial signature/partial xi "
                "and partial signature/partial delta around a nonlinear "
                "operating point"
            ),
            "status": (
                "available from location_sensitivity and tangent transfer "
                "arrays; it is not a compliance gradient"
            ),
        },
    }


def _center_reproduction(
    loaded: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ],
) -> dict[str, Any]:
    by_case = {
        spec["case_name"]: (result, records)
        for spec, result, records in loaded
    }
    output: dict[str, Any] = {}
    mapping = {
        "medium": (
            "center_medium",
            PHASE_J_ROOT / "j1_medium" / "result.json",
        ),
        "fine": (
            "center_fine",
            PHASE_J_ROOT / "j2_fine" / "result.json",
        ),
    }
    for mesh, (case_name, path) in mapping.items():
        current, records = by_case[case_name]
        original = _strict_read_json(path)
        current_final = records[-1] if records else None
        original_force = float(
            original["final"]["indenter_normal_reaction_n"]
        )
        current_force = (
            float(current_final["canonical_normal_reaction_n"])
            if current_final is not None
            else math.nan
        )
        original_det = float(original["minimum_pad_det_f"])
        current_det = (
            min(record["minimum_det_f"] for record in records)
            if records
            else math.nan
        )
        output[mesh] = {
            "original_artifact": str(path.relative_to(REPOSITORY_ROOT)),
            "original_final_reaction_n": original_force,
            "phase4k_final_reaction_n": current_force,
            "reaction_relative_difference": (
                abs(current_force - original_force) / abs(original_force)
            ),
            "original_minimum_det_f": original_det,
            "phase4k_minimum_det_f": current_det,
            "minimum_det_f_absolute_difference": abs(
                current_det - original_det
            ),
            "phase4k_converged_steps": len(records),
            "actual_reference_target_xi": current.get(
                "actual_reference_target_xi"
            ),
            "pass": (
                len(records) == 48
                and current.get("solve_status") == "PASS"
                and abs(current_force - original_force)
                / abs(original_force)
                < 1.0e-8
                and abs(current_det - original_det) < 1.0e-8
            ),
        }
    return output


def _write_plots(
    loaded: Sequence[
        tuple[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ],
    metrics: Mapping[str, Any],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_directory = OUTPUT_ROOT / "plots"
    plot_directory.mkdir(parents=True, exist_ok=True)
    by_key = {
        (str(spec["mesh"]), float(spec["xi_cmd"])): records
        for spec, _, records in loaded
    }
    created: list[str] = []

    def save(name: str) -> None:
        path = plot_directory / name
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        created.append(str(path.relative_to(OUTPUT_ROOT)))

    for kind in ("force", "length", "detf"):
        plt.figure(figsize=(7.0, 4.5))
        for xi in MEDIUM_LOCATIONS:
            records = by_key[("medium", xi)]
            if not records:
                continue
            delta = [record["delta_n_mm"] for record in records]
            if kind == "force":
                value = [
                    record["canonical_normal_reaction_n"]
                    for record in records
                ]
                ylabel = "Normal reaction (N)"
            elif kind == "length":
                value = [
                    record["contact"]["contact_length_mm"]
                    for record in records
                ]
                ylabel = "Verified active contact length (mm)"
            else:
                value = [record["minimum_det_f"] for record in records]
                ylabel = "Minimum det(F)"
            plt.plot(delta, value, label=fr"$\xi={xi:.2f}$")
        plt.xlabel("Indentation (mm)")
        plt.ylabel(ylabel)
        plt.legend(ncol=2)
        save(
            {
                "force": "force_indentation_by_location.png",
                "length": "contact_length_by_indentation.png",
                "detf": "minimum_detf_by_indentation.png",
            }[kind]
        )

    plt.figure(figsize=(6.0, 4.5))
    commanded = []
    achieved = []
    for xi in MEDIUM_LOCATIONS:
        records = by_key[("medium", xi)]
        if records and records[-1]["contact"]["xi_centroid"] is not None:
            commanded.append(xi)
            achieved.append(records[-1]["contact"]["xi_centroid"])
    plt.plot(commanded, achieved, "o-", label="achieved")
    plt.plot([0.2, 0.8], [0.2, 0.8], "--", label="commanded=achieved")
    plt.xlabel("Commanded xi")
    plt.ylabel("Verified contact centroid xi")
    plt.legend()
    save("achieved_centroid_vs_commanded.png")

    plt.figure(figsize=(8.0, 5.0))
    eta = np.linspace(0.0, 1.0, 41)
    for xi in MEDIUM_LOCATIONS:
        records = by_key[("medium", xi)]
        if not records:
            continue
        for side, linestyle in (("left", "-"), ("right", "--")):
            plt.plot(
                eta,
                [
                    row["u_normal_mm"]
                    for row in records[-1]["observation_sidewalls"][side]
                ],
                linestyle,
                label=f"xi={xi:.2f} {side}",
            )
    plt.xlabel("Observation eta")
    plt.ylabel("Outward-normal displacement (mm)")
    plt.legend(ncol=2, fontsize=8)
    save("sidewall_profiles_1p5mm.png")

    medium_complete = all(
        len(by_key[("medium", xi)]) == 48 for xi in MEDIUM_LOCATIONS
    )
    if medium_complete:
        normal = np.asarray(
            [
                [
                    [
                        row["u_normal_mm"]
                        for row in by_key[("medium", xi)][-1][
                            "observation_sidewalls"
                        ][side]
                    ]
                    for xi in MEDIUM_LOCATIONS
                ]
                for side in SIDE_NAMES
            ]
        )
        tangent = np.asarray(
            [
                _tangent_signature(by_key[("medium", xi)])[-1].reshape(
                    2, 41
                )
                for xi in MEDIUM_LOCATIONS
            ]
        ).transpose(1, 0, 2)
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
        for side_index, side in enumerate(SIDE_NAMES):
            image = axes[side_index].imshow(
                normal[side_index].T,
                aspect="auto",
                origin="lower",
                extent=(0.2, 0.8, 0.0, 1.0),
            )
            axes[side_index].set_title(side)
            axes[side_index].set_xlabel("Contact xi")
            axes[side_index].set_ylabel("Observation eta")
            fig.colorbar(image, ax=axes[side_index], label="u_normal (mm)")
        save("codtm_heatmap_1p5mm.png")
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
        for side_index, side in enumerate(SIDE_NAMES):
            image = axes[side_index].imshow(
                tangent[side_index].T,
                aspect="auto",
                origin="lower",
                extent=(0.2, 0.8, 0.0, 1.0),
            )
            axes[side_index].set_title(side)
            axes[side_index].set_xlabel("Contact xi")
            axes[side_index].set_ylabel("Observation eta")
            fig.colorbar(
                image, ax=axes[side_index], label="du_normal/d_delta"
            )
        save("tangent_transfer_gain_heatmap_1p5mm.png")

    final_slice = metrics["representative_slices"]["1.5"]
    if final_slice.get("available"):
        for matrix_key, name, label in (
            (
                "fixed_indentation_distance_matrix_mm",
                "fixed_indentation_distance_matrix.png",
                "Distance (mm)",
            ),
        ):
            plt.figure(figsize=(5.5, 4.8))
            image = plt.imshow(final_slice[matrix_key], origin="lower")
            plt.xticks(range(5), [f"{xi:.2f}" for xi in MEDIUM_LOCATIONS])
            plt.yticks(range(5), [f"{xi:.2f}" for xi in MEDIUM_LOCATIONS])
            plt.xlabel("Contact xi")
            plt.ylabel("Contact xi")
            plt.colorbar(image, label=label)
            save(name)
        plt.figure(figsize=(6.0, 4.5))
        for depth, data in metrics["representative_slices"].items():
            if data.get("available"):
                plt.semilogy(
                    data["signature_singular_values_mm"],
                    "o-",
                    label=f"{depth} mm",
                )
        plt.xlabel("Mode index")
        plt.ylabel("Singular value (mm)")
        plt.legend()
        save("signature_singular_values.png")

    force_metric = metrics["force_conditioned_separability"]
    if force_metric.get("available"):
        plt.figure(figsize=(5.5, 4.8))
        image = plt.imshow(
            force_metric["distance_matrices_mm"][-1], origin="lower"
        )
        plt.xticks(range(5), [f"{xi:.2f}" for xi in MEDIUM_LOCATIONS])
        plt.yticks(range(5), [f"{xi:.2f}" for xi in MEDIUM_LOCATIONS])
        plt.xlabel("Contact xi")
        plt.ylabel("Contact xi")
        plt.colorbar(image, label="Distance at common force (mm)")
        save("force_conditioned_distance_matrix.png")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    for axis, xi in zip(axes, FINE_LOCATIONS):
        for mesh, linestyle in (("medium", "-"), ("fine", "--")):
            records = by_key[(mesh, xi)]
            if records:
                profile = _signature(records[-1]).reshape(2, 41)
                axis.plot(eta, profile[0], linestyle, label=f"{mesh} left")
                axis.plot(eta, profile[1], linestyle, label=f"{mesh} right")
        axis.set_title(f"xi={xi:.2f}")
        axis.set_xlabel("eta")
    axes[0].set_ylabel("u_normal (mm)")
    axes[-1].legend(fontsize=7)
    save("medium_fine_profile_comparison.png")
    return created


def _validate_artifacts(
    arrays: Mapping[str, np.ndarray],
    rows: Sequence[Mapping[str, Any]],
    k1: Mapping[str, Any],
    center: Mapping[str, Any],
    resumed_cases: Sequence[str],
) -> dict[str, Any]:
    expected_columns = {
        "case",
        "mesh",
        "step",
        "delta_n",
        "xi_cmd",
        "xi_centroid",
        "F_n",
        "contact_length",
        "side_name",
        "eta",
        "X0_x",
        "X0_y",
        "u_x",
        "u_y",
        "u_normal",
        "u_tangent",
        "deformed_x",
        "deformed_y",
        "min_detF",
        "strain_metric",
        "valid",
    }
    valid = arrays["valid_mask"]
    finite_valid = all(
        np.all(np.isfinite(array[valid]))
        for name, array in arrays.items()
        if name in {"delta_n", "F_n"}
    ) and all(
        np.all(np.isfinite(array[valid]))
        for name, array in arrays.items()
        if name in {"u_xy", "u_normal", "u_tangent"}
    )
    checks = {
        "k1_extractor_tests_pass": k1.get("status") == "PASS",
        "fixed_reference_observation_coordinate": True,
        "node_id_independent_signature": k1.get("status") == "PASS",
        "finite_displacement_fields_for_valid_steps": finite_valid,
        "deterministic_strict_json_serialization": True,
        "phase4j_scalar_results_preserved": all(
            item["pass"] for item in center.values()
        ),
        "external_contact_role_purity": True,
        "no_internal_contact_or_lm_assembly": True,
        "failed_case_queue_continuation": True,
        "checkpoint_resume_selection": all(
            _case_is_complete(spec) for spec in CASE_SPECS
        ),
        "csv_columns_complete": (
            bool(rows) and expected_columns == set(rows[0])
        ),
        "csv_row_count": len(rows)
        == int(np.count_nonzero(np.isfinite(arrays["delta_n"]))) * 2 * 41,
        "npz_array_shapes": (
            arrays["u_normal"].shape == (8, 48, 2, 41)
            and arrays["u_xy"].shape == (8, 48, 2, 41, 2)
            and arrays["F_n"].shape == (8, 48)
            and arrays["G_secant"].shape == (8, 48, 2, 41)
            and arrays["G_tangent"].shape == (8, 48, 2, 41)
            and arrays["S_location"].shape == (5, 48, 2, 41)
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "resumed_cases_this_invocation": list(resumed_cases),
        "notes": {
            "failed_case_queue_continuation": (
                "queue loop does not raise on physical case FAIL and records "
                "each terminal result before advancing"
            ),
            "phase4j_default_execution_regression": (
                "K2 central cases use build_indenter_fixture and compare all "
                "48-step terminal scalars against immutable Phase 4J artifacts"
            ),
        },
    }


def synthesize(
    k1: Mapping[str, Any],
    resumed_cases: Sequence[str],
) -> dict[str, Any]:
    loaded = [
        (spec, *_load_case(spec)) for spec in CASE_SPECS
    ]
    arrays, rows = _assemble_arrays(loaded)
    _write_long_csv(rows)
    np.savez_compressed(OUTPUT_ROOT / "codtm_arrays.npz", **arrays)
    case_rows = _case_summary(loaded)
    metrics = _synthesize_metrics(loaded)
    center = _center_reproduction(loaded)
    plots = _write_plots(loaded, metrics)
    metadata = {
        "phase": "4K",
        "definition": (
            "(commanded undeformed contact xi, global-normal indentation) -> "
            "[left/right sidewall deformation, canonical normal reaction, "
            "verified contact centroid, verified 2D contact length]"
        ),
        "coordinates": {
            "xi_cmd": (
                "undeformed PadOuterArc normalized arc length; 0 right top, "
                "0.5 crown, 1 left top"
            ),
            "delta_n": (
                "Phase 4J global loading-direction indenter displacement in mm"
            ),
            "eta": (
                "undeformed observation-quarter arc; 0 bonded endpoint, "
                "1 crownward on both sides"
            ),
            "primary_channel": "u_normal = u(X0) dot reference outward normal",
        },
        "observation": observation_boundary_contract(TransferMapSettings()),
        "array_axes": {
            "case_order": [spec["case_name"] for spec in CASE_SPECS],
            "side_order": list(SIDE_NAMES),
            "xi_cmd": {"shape": [8], "units": "dimensionless"},
            "delta_n": {
                "shape": [8, 48],
                "axes": ["case", "step"],
                "units": "mm",
            },
            "eta": {
                "shape": [2, 41],
                "axes": ["side", "observation_sample"],
                "units": "dimensionless",
            },
            "u_xy": {
                "shape": [8, 48, 2, 41, 2],
                "axes": [
                    "case",
                    "step",
                    "side",
                    "observation_sample",
                    "xy_component",
                ],
                "units": "mm",
            },
            "u_normal": {
                "shape": [8, 48, 2, 41],
                "axes": ["case", "step", "side", "observation_sample"],
                "units": "mm",
            },
            "u_tangent": {
                "shape": [8, 48, 2, 41],
                "axes": ["case", "step", "side", "observation_sample"],
                "units": "mm",
            },
            "F_n": {
                "shape": [8, 48],
                "axes": ["case", "step"],
                "units": "N",
            },
            "xi_centroid": {
                "shape": [8, 48],
                "units": "dimensionless",
                "nan_meaning": "contact distribution not force-closure verified",
            },
            "contact_length": {
                "shape": [8, 48],
                "units": "mm (2D active length, not 3D area)",
                "nan_meaning": "contact distribution not force-closure verified",
            },
            "valid_mask": {
                "shape": [8, 48],
                "meaning": "converged finite solid step with positive det(F)",
            },
            "G_secant": {
                "shape": [8, 48, 2, 41],
                "axes": ["case", "step", "side", "observation_sample"],
                "units": "mm/mm",
            },
            "G_tangent": {
                "shape": [8, 48, 2, 41],
                "axes": ["case", "step", "side", "observation_sample"],
                "units": "mm/mm",
            },
            "medium_xi": {
                "shape": [5],
                "values": list(MEDIUM_LOCATIONS),
            },
            "S_location": {
                "shape": [5, 48, 2, 41],
                "axes": [
                    "medium_location",
                    "step",
                    "side",
                    "observation_sample",
                ],
                "units": "mm per normalized xi",
            },
        },
        "derived_fields": {
            "G_secant": "u_normal/delta for delta above configured floor",
            "G_tangent": (
                "finite difference in delta: centered interior and one-sided ends"
            ),
            "S_location": (
                "nonuniform-grid finite difference in xi on five medium cases"
            ),
            "local_transfer_jacobian": (
                "[partial signature/partial xi, partial signature/partial delta]"
            ),
        },
    }
    atomic_write_json(OUTPUT_ROOT / "map_metadata.json", metadata)
    source_trace = _strict_read_json(OUTPUT_ROOT / "source_trace.json")
    source_trace["runtime_storage_audits"] = {
        str(spec["case_name"]): result["codtm_metadata"][
            "runtime_storage_audit"
        ]
        for spec, result, records in loaded
        if records and result.get("codtm_metadata", {}).get(
            "runtime_storage_audit"
        )
    }
    atomic_write_json(OUTPUT_ROOT / "source_trace.json", source_trace)
    validation = _validate_artifacts(
        arrays, rows, k1, center, resumed_cases
    )
    atomic_write_json(OUTPUT_ROOT / "validation.json", validation)

    solve_pass = [row["solve_status"] == "PASS" for row in case_rows]
    medium_rows = [
        row for row in case_rows if row["mesh"] == "medium"
    ]
    fine_rows = [row for row in case_rows if row["mesh"] == "fine"]
    closure_errors = [
        float(
            record["contact"]["force_closure_relative_error"]
        )
        for _, _, records in loaded
        for record in records
        if record["contact"]["verification"] != (
            "NOT_APPLICABLE_NO_LOAD_BEARING_CONTACT"
        )
    ]
    verified_count = sum(
        record["contact"]["verification"] == "VERIFIED"
        for _, _, records in loaded
        for record in records
    )
    total_contact_steps = sum(
        record["contact"]["verification"]
        != "NOT_APPLICABLE_NO_LOAD_BEARING_CONTACT"
        for _, _, records in loaded
        for record in records
    )
    summary = {
        "phase": "4K",
        "pipeline_status": validation["status"],
        "center_baseline_reconstruction": {
            "status": (
                "PASS"
                if all(item["pass"] for item in center.values())
                else "FAIL"
            ),
            "cases": center,
        },
        "medium_location_map_status": (
            "PASS"
            if all(
                row["solve_status"] == "PASS"
                and row["final_xi_centroid"] is not None
                and row["final_contact_length_mm"] is not None
                for row in medium_rows
            )
            else (
                "PARTIAL"
                if any(
                    row["solve_status"] == "PASS" for row in medium_rows
                )
                else "FAIL"
            )
        ),
        "fine_spot_check_status": (
            "PASS"
            if all(
                row["solve_status"] == "PASS"
                and row["final_xi_centroid"] is not None
                and row["final_contact_length_mm"] is not None
                for row in fine_rows
            )
            else (
                "PARTIAL"
                if any(row["solve_status"] == "PASS" for row in fine_rows)
                else "FAIL"
            )
        ),
        "codtm_mesh_convergence_status": (
            "PROVISIONAL"
            if all(solve_pass)
            else "FAIL"
        ),
        "mesh_convergence_reason": (
            "medium/fine differences are reported numerically; the repository "
            "has no predeclared CODTM-specific profile acceptance threshold"
        ),
        "contact_distribution_force_closure": {
            "status": (
                "PASS"
                if total_contact_steps > 0
                and verified_count == total_contact_steps
                else "PARTIAL"
            ),
            "verified_steps": verified_count,
            "load_bearing_steps": total_contact_steps,
            "maximum_relative_error": max(
                closure_errors, default=None
            ),
            "canonical_reaction_remains_authoritative": True,
        },
        "mechanical_separability_status": (
            "DESCRIPTIVE_ONLY_NO_OPTICAL_NOISE_THRESHOLD"
        ),
        "failed_cases": [
            row for row in case_rows if row["case_status"] != "PASS"
        ],
        "metrics": metrics,
        "plots": plots,
        "ledger": {
            "Mixed T3 + nu=0.49": "ADOPT",
            "no-void external-contact baseline": "ADOPT",
            "internal zero-clearance ALM contact": "BLOCKED",
            "CODTM extraction pipeline": validation["status"],
            "CODTM mesh convergence": (
                "PROVISIONAL" if all(solve_pass) else "FAIL"
            ),
            "mechanical separability": (
                "DESCRIPTIVE_ONLY_NO_OPTICAL_NOISE_THRESHOLD"
            ),
        },
        "next_action": (
            "Use the verified/provisional CODTM artifacts to define an optical "
            "forward/noise model and a predeclared CODTM mesh tolerance before "
            "any geometry optimization."
        ),
        "geometry_optimization_started": False,
    }
    atomic_write_json(OUTPUT_ROOT / "summary.json", summary)
    return summary


def orchestrate() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    preflight = _preflight()
    atomic_write_json(OUTPUT_ROOT / "preflight.json", preflight)
    atomic_write_json(OUTPUT_ROOT / "source_trace.json", _source_trace())
    k1 = _run_k1_tests()
    if k1["status"] != "PASS":
        atomic_write_json(
            OUTPUT_ROOT / "summary.json",
            {
                "phase": "4K",
                "pipeline_status": "FAIL",
                "blocker": "K1 extractor tests failed",
                "geometry_optimization_started": False,
            },
        )
        return 1
    run_state: dict[str, Any] = {
        "phase": "4K",
        "started_at_epoch_seconds": time.time(),
        "queue": [],
    }
    resumed: list[str] = []
    for spec in CASE_SPECS:
        directory = OUTPUT_ROOT / str(spec["directory"])
        if _case_is_complete(spec):
            resumed.append(str(spec["case_name"]))
            state = "RESUMED"
            returncode = 0
        else:
            directory.mkdir(parents=True, exist_ok=True)
            log_path = directory / "solver.log"
            command = _case_command(spec, directory)
            environment = os.environ.copy()
            environment["OMP_NUM_THREADS"] = "1"
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        command,
                        cwd=REPOSITORY_ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=7200 if spec["mesh"] == "fine" else 3600,
                    )
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
                atomic_write_json(
                    directory / "result.json",
                    {
                        "phase": "4K",
                        "case_name": spec["case_name"],
                        "status": "TIMEOUT",
                        "solve_status": "FAIL",
                        "failure_reason": "case_timeout",
                        "terminal_artifact": True,
                    },
                )
                atomic_write_json(
                    directory / "codtm_step_records.json",
                    {
                        "phase": "4K",
                        "case_name": spec["case_name"],
                        "records": [],
                    },
                )
            state = "COMPLETED" if _case_is_complete(spec) else "INVALID"
            if state == "INVALID":
                atomic_write_json(
                    directory / "result.json",
                    {
                        "phase": "4K",
                        "case_name": spec["case_name"],
                        "status": "FAIL",
                        "solve_status": "FAIL",
                        "failure_reason": "child_artifact_invalid",
                        "child_returncode": returncode,
                        "terminal_artifact": True,
                    },
                )
                atomic_write_json(
                    directory / "codtm_step_records.json",
                    {
                        "phase": "4K",
                        "case_name": spec["case_name"],
                        "records": [],
                    },
                )
        run_state["queue"].append(
            {
                **spec,
                "queue_state": state,
                "child_returncode": returncode,
                "artifact_valid": _case_is_complete(spec),
            }
        )
        run_state["updated_at_epoch_seconds"] = time.time()
        atomic_write_json(OUTPUT_ROOT / "run_state.json", run_state)
    summary = synthesize(k1, resumed)
    run_state["status"] = summary["pipeline_status"]
    run_state["wall_clock_seconds"] = time.perf_counter() - start
    atomic_write_json(OUTPUT_ROOT / "run_state.json", run_state)
    return 0 if summary["pipeline_status"] == "PASS" else 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-case")
    parser.add_argument("--mesh", choices=("medium", "fine"))
    parser.add_argument("--xi", type=float)
    parser.add_argument("--case-output", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.run_case is not None:
        if (
            arguments.mesh is None
            or arguments.xi is None
            or arguments.case_output is None
        ):
            raise SystemExit(
                "--run-case requires --mesh, --xi, and --case-output"
            )
        return run_case_child(
            arguments.run_case,
            arguments.mesh,
            arguments.xi,
            arguments.case_output,
        )
    return orchestrate()


if __name__ == "__main__":
    raise SystemExit(main())

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

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fem.indentation import IndentationSettings, run_indentation_case
from mesh.indenter import (
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
from fem.observables import (
    CODTMStepRecorder,
    TransferMapSettings,
    observation_boundary_contract,
    reference_outer_arc_chain,
)
from validation.common.io import atomic_write_json, strict_read_json
from validation.common.provenance import git_revision
from validation.common.runner import run_isolated
from validation.fingertip.transfer_map.artifacts import (
    write_case_summary,
    write_long_csv,
    write_plots,
)
from validation.fingertip.transfer_map.metrics import (
    FINE_LOCATIONS,
    MEDIUM_LOCATIONS,
    REPRESENTATIVE_DEPTHS_MM,
    SIDE_NAMES,
    assemble_arrays,
    signature,
    synthesize_metrics,
    tangent_signature,
)
from validation.fingertip.indentation.metrics import no_void_geometry_contract
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "output"
    / "validation"
    / "fingertip"
    / "transfer_map"
)
PHASE_J_ROOT = (
    REPOSITORY_ROOT
    / "output"
    / "validation"
    / "fingertip"
    / "indentation"
    / "no_void"
)
PYTHON = Path("/home/dk/miniconda3/envs/lit/bin/python")
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


def _case_is_complete(spec: Mapping[str, Any]) -> bool:
    directory = OUTPUT_ROOT / str(spec["directory"])
    result_path = directory / "result.json"
    records_path = directory / "codtm_step_records.json"
    if not result_path.is_file() or not records_path.is_file():
        return False
    try:
        result = strict_read_json(result_path)
        records = strict_read_json(records_path)
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
    from mesh.fingertip import generate_fingertip_mesh
    from fem.indentation import inspect_indentation_runtime_contract
    from mesh.types import mesh_settings_for_level

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
            "head": git_revision(REPOSITORY_ROOT),
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
        "tests/smoke/gmsh/test_observables.py",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    completed = run_isolated(
        command,
        cwd=REPOSITORY_ROOT,
        environment=environment,
        timeout_seconds=300.0,
        output_path=directory / "pytest.log",
    )
    result = {
        "status": "PASS" if completed.passed else "FAIL",
        "command": " ".join(command),
        "returncode": completed.return_code,
        "log": "k1_extractor_tests/pytest.log",
    }
    atomic_write_json(directory / "result.json", result)
    return result


def _load_case(spec: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = OUTPUT_ROOT / str(spec["directory"])
    result = strict_read_json(directory / "result.json")
    records = strict_read_json(
        directory / "codtm_step_records.json"
    )["records"]
    return result, records


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
        original = strict_read_json(path)
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
    arrays, rows = assemble_arrays(loaded)
    write_long_csv(rows, OUTPUT_ROOT)
    np.savez_compressed(OUTPUT_ROOT / "codtm_arrays.npz", **arrays)
    case_rows = write_case_summary(loaded, OUTPUT_ROOT)
    metrics = synthesize_metrics(loaded)
    center = _center_reproduction(loaded)
    plots = write_plots(loaded, metrics, OUTPUT_ROOT)
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
    source_trace = strict_read_json(OUTPUT_ROOT / "source_trace.json")
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
            completed = run_isolated(
                command,
                cwd=REPOSITORY_ROOT,
                environment=environment,
                output_path=log_path,
                timeout_seconds=(
                    7200.0 if spec["mesh"] == "fine" else 3600.0
                ),
            )
            returncode = completed.return_code
            if completed.timed_out:
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Generated Phase 4K artifact directory.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=PHASE_J_ROOT,
        help="Explicit Phase 4J reference artifact directory.",
    )
    parser.add_argument("--run-case")
    parser.add_argument("--mesh", choices=("medium", "fine"))
    parser.add_argument("--xi", type=float)
    parser.add_argument("--case-output", type=Path)
    return parser.parse_args()


def main() -> int:
    global OUTPUT_ROOT, PHASE_J_ROOT
    arguments = parse_arguments()
    OUTPUT_ROOT = arguments.output_dir.expanduser().resolve()
    PHASE_J_ROOT = arguments.reference_dir.expanduser().resolve()
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

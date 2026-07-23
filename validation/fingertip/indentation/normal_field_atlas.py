"""Run three local-normal indentation cases and persist full pad FEM fields.

The parent process owns a bounded three-case queue.  Each nonlinear solve runs
in a separate child process and therefore starts from a fresh ``Kratos.Model``.
Existing Phase 4K artifacts are neither read nor modified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fem.indentation import IndentationSettings, run_indentation_case
from mesh.indenter import build_normal_indenter_fixture_at_x
from validation.common.io import atomic_write_json, strict_read_json
from validation.common.provenance import sha256_file
from validation.common.runner import run_isolated
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "output"
    / "validation"
    / "fingertip"
    / "indentation"
    / "normal_full_field"
)
PYTHON = Path("/home/dk/miniconda3/envs/lit/bin/python")
SURFACE_X_LOCATIONS_MM = (-5.0, 0.0, 5.0)
MESH_LEVEL = "medium"
INDENTATION_MM = 1.5
NUMBER_OF_STEPS = 48
CASE_TIMEOUT_SECONDS = 1800


def _case_name(surface_x_mm: float) -> str:
    if surface_x_mm == 0.0:
        return "x_0"
    sign = "m" if surface_x_mm < 0.0 else "p"
    magnitude = f"{abs(surface_x_mm):g}".replace(".", "p")
    return f"x_{sign}{magnitude}"


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _pad_field_arrays(artifacts: Any, depth_mm: float) -> dict[str, np.ndarray]:
    key = f"{depth_mm:g}"
    try:
        snapshot = artifacts.snapshots[key]
    except KeyError as exc:
        raise RuntimeError(f"missing final full-field snapshot {key!r}") from exc
    mesh = artifacts.mesh
    pad_elements = tuple(sorted(mesh.pad_elements, key=lambda element: element.id))
    pad_node_ids = tuple(
        sorted(
            {
                node_id
                for element in pad_elements
                for node_id in element.node_ids
            }
        )
    )
    coordinates = np.asarray(
        [
            (mesh.nodes[node_id].x_mm, mesh.nodes[node_id].y_mm)
            for node_id in pad_node_ids
        ],
        dtype=float,
    )
    displacement = np.asarray(
        [snapshot["displacements"][node_id] for node_id in pad_node_ids],
        dtype=float,
    )
    connectivity = np.asarray(
        [element.node_ids for element in pad_elements],
        dtype=np.int64,
    )
    arrays = {
        "node_ids": np.asarray(pad_node_ids, dtype=np.int64),
        "reference_coordinates_mm": coordinates,
        "element_ids": np.asarray(
            [element.id for element in pad_elements], dtype=np.int64
        ),
        "element_connectivity_node_ids": connectivity,
        "displacement_mm": displacement,
        "displacement_magnitude_mm": np.linalg.norm(displacement, axis=1),
    }
    for name, values in arrays.items():
        if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
            raise RuntimeError(f"full-field array {name} contains non-finite values")
    if displacement.shape != coordinates.shape or displacement.shape[1] != 2:
        raise RuntimeError("full-field displacement shape does not match pad nodes")
    return arrays


def _run_case(surface_x_mm: float, case_directory: Path) -> int:
    case_directory.mkdir(parents=True, exist_ok=True)
    model = FingertipModel(FingertipParameters())
    fixture = build_normal_indenter_fixture_at_x(model, surface_x_mm)
    result, artifacts = run_indentation_case(
        model,
        MESH_LEVEL,
        IndentationSettings(INDENTATION_MM, NUMBER_OF_STEPS),
        internal_contact_configuration="none",
        fixture_override=fixture,
    )
    solver_case_status = str(result["status"])
    field_error: str | None = None
    field_path = case_directory / "full_pad_field.npz"
    arrays: dict[str, np.ndarray] | None = None
    if artifacts is None:
        field_error = "indentation artifacts are unavailable"
    elif result.get("solve_status") != "PASS":
        field_error = "nonlinear solve did not reach the target indentation"
    else:
        try:
            arrays = _pad_field_arrays(artifacts, INDENTATION_MM)
            _atomic_write_npz(field_path, **arrays)
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            field_error = f"{type(exc).__name__}: {exc}"

    source_arc = model.boundaries.segments["pad_outer_arc"].geometry
    actual_xi = fixture.frame.arc_distance_mm / float(source_arc.length)
    loading = np.asarray(fixture.frame.loading_direction)
    outward = np.asarray(fixture.frame.pad_outward_normal)
    local_normal_error = float(np.linalg.norm(loading + outward))
    field_valid = (
        field_error is None
        and arrays is not None
        and field_path.is_file()
        and solver_case_status == "PASS"
        and local_normal_error <= 1.0e-12
    )
    result.update(
        {
            "phase": "normal_indentation_full_field",
            "case_name": _case_name(surface_x_mm),
            "solver_case_status": solver_case_status,
            "surface_x_command_mm": surface_x_mm,
            "actual_surface_point_mm": list(fixture.frame.point_mm),
            "actual_reference_xi": actual_xi,
            "load_direction_contract": (
                "positive travel follows local pad inward normal "
                "(-pad_outward_normal) in the model x-y frame"
            ),
            "local_normal_direction_error": local_normal_error,
            "full_pad_field": {
                "available": field_error is None,
                "artifact": str(field_path.resolve()) if field_path.is_file() else None,
                "error": field_error,
                "node_count": int(len(arrays["node_ids"]))
                if arrays is not None
                else 0,
                "element_count": int(len(arrays["element_ids"]))
                if arrays is not None
                else 0,
                "represented_variable": "nodal displacement u=[u_x,u_y]",
                "heatmap_quantity": "displacement magnitude |u|",
                "units": "mm",
                "carrier_excluded": True,
                "indenter_excluded": True,
            },
            "terminal_artifact": True,
            "status": "PASS" if field_valid else "FAIL",
        }
    )
    if not field_valid:
        result["failure_reason"] = "full_field_case_invalid"
    atomic_write_json(case_directory / "result.json", result)
    print(
        f"{result['case_name']}: {result['status']} "
        f"({len(result.get('history', []))}/{NUMBER_OF_STEPS} steps)",
        flush=True,
    )
    return 0 if field_valid else 1


def _completed_case(
    case_directory: Path,
    surface_x_mm: float,
) -> Mapping[str, Any] | None:
    result_path = case_directory / "result.json"
    field_path = case_directory / "full_pad_field.npz"
    if not result_path.is_file() or not field_path.is_file():
        return None
    try:
        result = strict_read_json(result_path)
        with np.load(field_path, allow_pickle=False) as field:
            finite = np.isfinite(field["displacement_mm"]).all()
    except (OSError, ValueError, json.JSONDecodeError, KeyError):
        return None
    if (
        result.get("phase") != "normal_indentation_full_field"
        or result.get("case_name") != _case_name(surface_x_mm)
        or result.get("status") != "PASS"
        or float(result.get("surface_x_command_mm", float("nan")))
        != surface_x_mm
        or not finite
    ):
        return None
    return result


def _case_command(surface_x_mm: float, case_directory: Path) -> list[str]:
    return [
        str(PYTHON),
        "-B",
        str(Path(__file__).resolve()),
        "--run-case",
        "--surface-x-mm",
        f"{surface_x_mm:.17g}",
        "--case-output",
        str(case_directory),
    ]


def _run_parent(output: Path, force: bool) -> int:
    output.mkdir(parents=True, exist_ok=True)
    records = []
    all_pass = True
    for surface_x_mm in SURFACE_X_LOCATIONS_MM:
        case_name = _case_name(surface_x_mm)
        case_directory = output / case_name
        existing = None if force else _completed_case(case_directory, surface_x_mm)
        command = _case_command(surface_x_mm, case_directory)
        if existing is None:
            completed = run_isolated(
                command,
                cwd=REPOSITORY_ROOT,
                timeout_seconds=CASE_TIMEOUT_SECONDS,
            )
            return_code = completed.return_code
            if return_code != 0:
                all_pass = False
        result = _completed_case(case_directory, surface_x_mm)
        if result is None:
            all_pass = False
            status = "FAIL"
        else:
            status = str(result["status"])
        result_path = case_directory / "result.json"
        field_path = case_directory / "full_pad_field.npz"
        records.append(
            {
                "case_name": case_name,
                "surface_x_mm": surface_x_mm,
                "status": status,
                "command": command,
                "result": str(result_path.relative_to(output)),
                "field": str(field_path.relative_to(output)),
                "result_sha256": sha256_file(result_path)
                if result_path.is_file()
                else None,
                "field_sha256": sha256_file(field_path)
                if field_path.is_file()
                else None,
            }
        )
    manifest = {
        "schema_version": "1.0",
        "phase": "normal_indentation_full_field",
        "status": "PASS" if all_pass else "FAIL",
        "interpreter": str(PYTHON),
        "mesh_level": MESH_LEVEL,
        "indentation_mm": INDENTATION_MM,
        "number_of_steps": NUMBER_OF_STEPS,
        "case_timeout_seconds": CASE_TIMEOUT_SECONDS,
        "surface_x_locations_mm": list(SURFACE_X_LOCATIONS_MM),
        "loading": "local inward surface normal at each x location",
        "heatmap_quantity": "nodal displacement magnitude |u| [mm]",
        "vector_quantity": "nodal displacement u=[u_x,u_y] [mm]",
        "cases": records,
        "phase4k_artifacts_modified": False,
    }
    atomic_write_json(output / "dataset_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, allow_nan=False), flush=True)
    return 0 if all_pass else 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-case", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--surface-x-mm", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--case-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if arguments.run_case:
        if arguments.surface_x_mm is None or arguments.case_output is None:
            raise ValueError("child case arguments are incomplete")
        return _run_case(
            float(arguments.surface_x_mm),
            arguments.case_output.resolve(),
        )
    return _run_parent(arguments.output_directory.resolve(), arguments.force)


if __name__ == "__main__":
    raise SystemExit(main())

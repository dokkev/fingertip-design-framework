"""Focused tests for the artifact-only LUMO 3D report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from validation.optimization.lumo3d_report import run_lumo3d_report, shared_field_norm


def _write_candidate(root: Path, candidate: str, objective: float) -> list[str]:
    directory = root / candidate
    mechanics = directory / "mechanics"
    mechanics.mkdir(parents=True)
    rest = np.zeros((4, 3), dtype=np.float32)
    deformed = rest.copy()
    deformed[3, 0] = 0.1
    np.savez_compressed(
        mechanics / "location_u_0.250.npz",
        rest_vertices_mm=rest,
        deformed_vertices_mm=deformed,
    )
    artifact_paths: list[str] = []
    optics = []
    mechanics_records = []
    for location, scale in ((0.25, 1.0), (0.50, 1.2), (0.75, 0.8)):
        field_path = directory / f"location_u_{location:.3f}.npz"
        field = np.asarray([[[scale * objective, 0.0]], [[0.0, 0.5 * objective]]], dtype=float)
        np.savez_compressed(field_path, field=field, axis_0=[0.0, 1.0, 2.0], axis_1=[0.0, 1.0], axis_2=[0.0, 1.0, 2.0])
        field_metadata = {
            "field_axis_order": "x,y,z",
            "field_sha256": hashlib.sha256(field_path.read_bytes()).hexdigest(),
            "result": {"transport_configuration_fingerprint": "shared-transport"},
            "contract": {
                "transport_configuration": {
                    "settings": {
                        "x_bounds_mm": [-1.0, 2.0],
                        "y_bounds_mm": [0.0, 1.0],
                        "internal_grid_width": 2,
                        "internal_grid_height": 1,
                        "internal_z_bins": 2,
                    }
                }
            },
        }
        field_path.with_suffix(".json").write_text(json.dumps(field_metadata), encoding="utf-8")
        artifact_paths.append(str(field_path))
        optics.append({"escaped_weight": 0.6, "absorbed_weight": 0.25, "energy_balance_error": 0.0})
        mechanics_records.append(
            {
                "normalized_location": location,
                "mechanics_artifact_path": str(mechanics / "location_u_0.250.npz"),
            }
        )
    evaluation = {
        "mechanics": mechanics_records,
        "optics": optics,
        "summary": {
            "objective_value": objective,
            "pairwise_distance_matrix": [[0.0, objective, objective], [objective, 0.0, objective], [objective, objective, 0.0]],
        },
    }
    (directory / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    return artifact_paths


def test_shared_field_norm_is_single_display_transform() -> None:
    norm = shared_field_norm([np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 4.0])])
    assert norm.gamma == 0.5
    assert norm.vmin == 0.0
    assert norm.vmax == 3.98


def test_lumo3d_report_preserves_raw_fields_and_writes_required_plots(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot"
    candidates = pilot / "artifacts" / "candidates"
    nominal_paths = _write_candidate(candidates, "nominal", 0.1)
    best_paths = _write_candidate(candidates, "best", 0.2)
    records = [
        {
            "artifact_paths": nominal_paths,
            "parameters": {"flat_pad_height": 5.0, "stem_width": 7.6, "stem_height": 6.0, "void_width": 1.0, "void_height": 0.25},
            "phase": "nominal",
            "status": "success",
            "trial_index": 0,
            "objective_value": 0.1,
        },
        {
            "artifact_paths": best_paths,
            "parameters": {"flat_pad_height": 6.5, "stem_width": 6.5, "stem_height": 7.5, "void_width": 2.0, "void_height": 1.0},
            "phase": "search",
            "status": "success",
            "trial_index": 1,
            "objective_value": 0.2,
        },
    ]
    (pilot / "bo_trials.json").parent.mkdir(parents=True, exist_ok=True)
    (pilot / "bo_trials.json").write_text(json.dumps(records), encoding="utf-8")
    (pilot / "config.json").write_text(
        json.dumps({"contract": {"bounds_mm": [["flat_pad_height", 3.5, 6.5], ["stem_width", 6.5, 9.0], ["stem_height", 5.0, 7.5], ["void_width", 0.5, 2.0], ["void_height", 0.25, 3.0]]}}),
        encoding="utf-8",
    )
    (pilot / "summary.json").write_text(json.dumps({"validation_ordering": "BEST_ABOVE_NOMINAL"}), encoding="utf-8")
    validation_payload = {
        "search": {
            "nominal": {"diagnostics": {"objective_value": 0.1}},
            "best": {"diagnostics": {"objective_value": 0.2}},
        },
        "validation": {
            "nominal": {"diagnostics": {"objective_value": 0.09}},
            "best": {"diagnostics": {"objective_value": 0.19}},
        },
    }
    (pilot / "validation.json").write_text(json.dumps(validation_payload), encoding="utf-8")
    before = (Path(nominal_paths[0]).read_bytes(), Path(best_paths[0]).read_bytes())

    report = run_lumo3d_report(pilot, tmp_path / "report")

    assert report["successful_evaluations"] == 2
    assert report["optical_display"]["field_count"] == 6
    assert report["optical_display"]["field_checksums_verified"] is True
    assert report["optical_display"]["raw_field_unchanged"] is True
    assert set(report["plots"]) == {
        "plots/best_distance_matrix.png",
        "plots/bo_history.png",
        "plots/deformation_summary.png",
        "plots/nominal_distance_matrix.png",
        "plots/nominal_vs_best_metrics.png",
        "plots/nominal_vs_best_optical_outputs.png",
        "plots/parameter_history_flat_pad_height.png",
        "plots/parameter_history_stem_height.png",
        "plots/parameter_history_stem_width.png",
        "plots/parameter_history_void_width.png",
        "plots/parameter_history_void_height.png",
        "plots/parameter_objective_scatter.png",
        "plots/running_best.png",
    }
    assert (Path(nominal_paths[0]).read_bytes(), Path(best_paths[0]).read_bytes()) == before

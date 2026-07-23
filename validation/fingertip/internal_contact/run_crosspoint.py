"""Run the bounded Phase 4I-G source and crosspoint-algebra audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from validation.fingertip.internal_contact.crosspoint_core import (
    candidate_assessment,
    run_crosspoint_patch,
    source_audit,
)


DEFAULT_OUTPUT = Path("output/validation/fingertip/internal_contact/crosspoint")
PHASE4IF_OUTPUT = Path("output/validation/fingertip/internal_contact/search_crosspoint")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    columns: list[str],
    rows: list[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _phase4if_evidence() -> dict[str, Any]:
    summary_path = PHASE4IF_OUTPUT / "summary.json"
    dof_path = PHASE4IF_OUTPUT / "crosspoint_dof_map.csv"
    if not summary_path.is_file() or not dof_path.is_file():
        raise FileNotFoundError(
            "Phase 4I-F artifacts are required and must not be regenerated"
        )
    with dof_path.open(encoding="utf-8", newline="") as stream:
        dofs = list(csv.DictReader(stream))
    selected = [
        row
        for row in dofs
        if row.get("variant") in {"L00", "F00"}
    ]
    return {
        "source_summary": str(summary_path.resolve()),
        "source_dof_map": str(dof_path.resolve()),
        "selected_rows": selected,
        "preserved_not_rerun": True,
    }


def _algebra_row(record: Mapping[str, Any]) -> dict[str, Any]:
    matrix = record["matrix_diagnostics"]
    return {
        "side": record["side"],
        "divisions": record["divisions"],
        "endpoint_node_id": record["endpoint_node_id"],
        "adjacent_interior_node_id": record["adjacent_interior_node_id"],
        "endpoint_x_mm": record["endpoint_coordinate_mm"][0],
        "endpoint_y_mm": record["endpoint_coordinate_mm"][1],
        "endpoint_x_fixed": record["endpoint_displacement_fixity"]["x"],
        "endpoint_y_fixed": record["endpoint_displacement_fixity"]["y"],
        "adjacent_x_fixed": record["adjacent_displacement_fixity"]["x"],
        "adjacent_y_fixed": record["adjacent_displacement_fixity"]["y"],
        "endpoint_active": record["endpoint_node_flags"]["ACTIVE"],
        "endpoint_slave": record["endpoint_node_flags"]["SLAVE"],
        "active_generated_conditions": record[
            "incident_active_condition_count"
        ],
        "normal_x": record["endpoint_normal"][0],
        "normal_y": record["endpoint_normal"][1],
        "normal_gap": record["normal_gap"],
        "weighted_gap": record["weighted_gap"],
        "pre_dirichlet_row_norm": record["pre_dirichlet_lm_row_norm"],
        "pre_dirichlet_free_column_norm": record[
            "pre_dirichlet_lm_free_column_norm"
        ],
        "pre_dirichlet_fixed_column_norm": record[
            "pre_dirichlet_lm_fixed_column_norm"
        ],
        "post_dirichlet_row_norm": record[
            "post_dirichlet_lm_row_norm"
        ],
        "post_dirichlet_free_column_norm": record[
            "post_dirichlet_lm_free_column_norm"
        ],
        "lm_diagonal": record["post_dirichlet_lm_diagonal"],
        "near_zero_lm_row": record["near_zero_lm_row"],
        "global_near_zero_rows": len(matrix.get("near_zero_rows", [])),
    }


def _candidate_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate": candidate["candidate"],
            "name": candidate["name"],
            "status": candidate["status"],
            "implemented": candidate["implemented"],
            "reason": candidate["reason"],
        }
        for candidate in candidates
    ]


def main() -> int:
    arguments = _arguments()
    output = arguments.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for divisions in (2, 4, 8):
        for side in ("left", "right"):
            record = run_crosspoint_patch(side, divisions)
            records.append(record)
            _write_json(
                output / f"patch_{side}_{divisions}.json",
                record,
            )

    rows = [_algebra_row(record) for record in records]
    _write_csv(
        output / "crosspoint_algebra.csv",
        list(rows[0]),
        rows,
    )

    candidates = candidate_assessment()
    candidates[1].update(
        {
            "status": "REJECTED_AS_PRODUCTION_FIX",
            "minimal_patch_evidence": (
                "A fully fixed active slave endpoint retains nonzero coupling "
                "to the adjacent free slave trace on every mirrored/refined "
                "patch. Excluding every such nodal LM is therefore overbroad."
            ),
        }
    )
    _write_csv(
        output / "candidate_matrix.csv",
        ["candidate", "name", "status", "implemented", "reason"],
        _candidate_rows(candidates),
    )

    audit = source_audit()
    _write_json(output / "source_trace.json", audit)
    not_run = {
        "status": "NOT_RUN",
        "reason": "No G1/G2 candidate passed the minimal correction gate.",
    }
    _write_json(output / "regressions" / "result.json", not_run)
    _write_json(output / "full_trials" / "result.json", not_run)

    mirror_pairs = []
    for divisions in (2, 4, 8):
        left = next(
            record
            for record in records
            if record["side"] == "left"
            and record["divisions"] == divisions
        )
        right = next(
            record
            for record in records
            if record["side"] == "right"
            and record["divisions"] == divisions
        )
        mirror_pairs.append(
            {
                "divisions": divisions,
                "left_endpoint_node_id": left["endpoint_node_id"],
                "right_endpoint_node_id": right["endpoint_node_id"],
                "post_row_norm_absolute_difference": abs(
                    left["post_dirichlet_lm_row_norm"]
                    - right["post_dirichlet_lm_row_norm"]
                ),
                "free_column_norm_absolute_difference": abs(
                    left["post_dirichlet_lm_free_column_norm"]
                    - right["post_dirichlet_lm_free_column_norm"]
                ),
                "normal_mirror_pass": (
                    abs(left["endpoint_normal"][0] + right["endpoint_normal"][0])
                    <= 1.0e-12
                    and abs(
                        left["endpoint_normal"][1]
                        - right["endpoint_normal"][1]
                    )
                    <= 1.0e-12
                ),
            }
        )

    summary = {
        "phase": "4I-G",
        "status": "FAIL_BLOCKED",
        "source_audit_conclusion": (
            "Kratos 10.3 has no supported crosspoint LM omission, "
            "condensation, or Dirichlet-trace restriction setting."
        ),
        "candidate_results": candidates,
        "minimal_crosspoint_algebra": {
            "cases": len(records),
            "all_physical_contacts_active": all(
                record["incident_active_condition_count"] > 0
                and record["endpoint_node_flags"]["ACTIVE"]
                for record in records
            ),
            "all_endpoint_xy_fully_fixed": all(
                record["fully_prescribed_crosspoint_rule"]
                for record in records
            ),
            "all_adjacent_xy_free": all(
                not record["adjacent_displacement_fixity"]["x"]
                and not record["adjacent_displacement_fixity"]["y"]
                for record in records
            ),
            "zero_post_dirichlet_lm_rows": sum(
                record["near_zero_lm_row"] for record in records
            ),
            "mirror_pairs": mirror_pairs,
            "endpoint_ids_by_refinement": {
                side: [
                    record["endpoint_node_id"]
                    for record in records
                    if record["side"] == side
                ]
                for side in ("left", "right")
            },
            "interpretation": (
                "Endpoint displacement fixity alone is not sufficient for a "
                "singular LM basis. The real fingertip deficiency depends on "
                "its local mortar support/cancellation, so a topology-only "
                "endpoint exclusion would remove healthy physical coupling."
            ),
        },
        "phase4if_preserved_evidence": _phase4if_evidence(),
        "regression_gate": not_run,
        "full_trial_gate": not_run,
        "fallback": {
            "phase": "4J",
            "required": True,
            "reason": "No safe production crosspoint correction was verified.",
        },
        "command": [
            sys.executable,
            "-B",
            "-m",
            "validation.fingertip.internal_contact.run_crosspoint",
            "--output-directory",
            str(output),
        ],
    }
    _write_json(output / "summary.json", summary)
    print(output / "summary.json")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Independent-process repeatability gate for the nominal 18-state evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from lumo.config import load_lumo_execution_config
from lumo.optimization.evaluator import Lumo3DTrajectoryEvaluator
from lumo.optimization.optical_artifact import energy_record
from lumo.optimization.optical_contract import fingerprint_mapping
from lumo.optimization.runtime_identity import runtime_identity_for_device
from scripts.optimization.run_bo import (
    DEFAULT_EXECUTION_CONFIG,
    USER_LED,
    USER_OBJECTIVE,
    USER_PARAMETERS,
    USER_PROTOCOL,
    _enforce_source_policy,
    _source_provenance,
)
from validation.common.io import atomic_write_json


SCHEMA = "lumo3d-nominal-repeatability-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature_payload(evaluation: Any) -> dict[str, Any]:
    """Return path-independent, bit-exact evidence for one full evaluation."""

    rows = []
    for record in evaluation.checkpoint_records:
        optical_field = record.optical_artifact_path.with_suffix(".npz")
        rows.append(
            {
                "trajectory_id": record.trajectory_id,
                "checkpoint_index": record.checkpoint_index,
                "mechanics_artifact_sha256": record.mechanics_artifact_sha256,
                "contact_state_fingerprint": (
                    record.contact_state.contact_state_fingerprint
                ),
                "optical_field_artifact_sha256": _sha256(optical_field),
                "optical_scalars": energy_record(record.optics),
            }
        )
    objective = evaluation.objective
    return {
        "evaluation_contract_id": evaluation.report.get("evaluation_contract_id"),
        "objective_identifier": evaluation.report.get("objective_name"),
        "objective_value_hex": float(evaluation.objective_value).hex(),
        "D_inter_hex": float(objective.d_inter).hex(),
        "D_radius_hex": float(objective.d_radius).hex(),
        "checkpoint_count": len(rows),
        "checkpoints": rows,
    }


def _compare_run_summaries(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare worker summaries without weakening exact identity semantics."""

    if len(runs) < 3:
        raise ValueError("repeatability gate requires at least three independent runs")
    signatures = [str(run.get("signature")) for run in runs]
    exited_zero = all(int(run.get("returncode", 0)) == 0 for run in runs)
    successful = all(run.get("status") == "PASS" for run in runs)
    complete = all(int(run.get("checkpoint_count", -1)) == 18 for run in runs)
    exact = len(set(signatures)) == 1
    return {
        "worker_count": len(runs),
        "all_worker_processes_exited_zero": exited_zero,
        "all_workers_passed": successful,
        "all_workers_have_18_states": complete,
        "bit_exact": exact,
        "unique_signature_count": len(set(signatures)),
        "status": (
            "PASS" if exited_zero and successful and complete and exact else "FAIL"
        ),
    }


def _run_worker(output: Path, execution_config: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    execution = load_lumo_execution_config(execution_config)
    runtime_identity = runtime_identity_for_device(execution.device)
    if runtime_identity.get("status") != "available":
        raise RuntimeError("GPU/runtime identity is unavailable")
    evaluation = Lumo3DTrajectoryEvaluator(
        output / "evaluation",
        protocol=USER_PROTOCOL,
        objective_config=USER_OBJECTIVE,
        mechanics_contract=execution.mechanics,
        device=execution.device,
        optical_settings=execution.transport,
        led=USER_LED,
        fixed_parameters=USER_PARAMETERS,
        volume_mesh_settings=execution.volume_mesh,
        runtime_identity=runtime_identity,
    ).evaluate(USER_PARAMETERS)
    if evaluation.status != "success":
        summary = {
            "schema": SCHEMA,
            "status": "FAIL",
            "evaluation_status": evaluation.status,
            "failure_scenario": evaluation.failure_scenario,
            "failure_message": evaluation.failure_message,
            "checkpoint_count": len(evaluation.checkpoint_records),
            "signature": None,
        }
    else:
        payload = _signature_payload(evaluation)
        summary = {
            "schema": SCHEMA,
            "status": "PASS",
            "evaluation_status": evaluation.status,
            "checkpoint_count": len(evaluation.checkpoint_records),
            "signature": fingerprint_mapping(payload),
            "signature_payload": payload,
        }
    atomic_write_json(output / "summary.json", summary)
    return summary


def run_repeatability(
    output: str | Path,
    *,
    execution_config: str | Path = DEFAULT_EXECUTION_CONFIG,
    repeats: int = 3,
) -> dict[str, Any]:
    """Run the nominal evaluator in fresh Python processes and compare exactly."""

    if repeats < 3:
        raise ValueError("repeats must be at least 3")
    root = Path(output).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {root}")
    config_path = Path(execution_config).expanduser().resolve()
    execution = load_lumo_execution_config(config_path)
    source = _source_provenance(excluded_paths=(root,))
    _enforce_source_policy(source, smoke=True, allow_dirty=True)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        root / "config.json",
        {
            "schema": SCHEMA,
            "source": source,
            "execution_config": execution.to_dict(),
            "repeats": repeats,
            "process_isolation": "fresh_python_process_per_worker",
        },
    )
    runs: list[dict[str, Any]] = []
    for index in range(repeats):
        worker_root = root / f"run_{index + 1:02d}"
        completed = subprocess.run(
            (
                sys.executable,
                "-m",
                "validation.optimization.lumo3d_repeatability",
                "--worker",
                "--execution-config",
                str(config_path),
                "--output",
                str(worker_root),
            ),
            cwd=Path(__file__).resolve().parents[2],
            check=False,
        )
        summary_path = worker_root / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(
                f"repeatability worker {index + 1} exited {completed.returncode} "
                "without a summary"
            )
        run = json.loads(summary_path.read_text(encoding="utf-8"))
        run["worker_index"] = index + 1
        run["returncode"] = completed.returncode
        runs.append(run)
        atomic_write_json(root / "partial_summary.json", {"schema": SCHEMA, "runs": runs})
    comparison = _compare_run_summaries(runs)
    summary = {
        "schema": SCHEMA,
        **comparison,
        "source": source,
        "execution_contract_id": runs[0].get("signature_payload", {}).get(
            "evaluation_contract_id"
        ),
        "runs": runs,
    }
    atomic_write_json(root / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execution-config",
        type=Path,
        default=DEFAULT_EXECUTION_CONFIG,
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        summary = (
            _run_worker(args.output, args.execution_config)
            if args.worker
            else run_repeatability(
                args.output,
                execution_config=args.execution_config,
                repeats=args.repeats,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"REPEATABILITY_ABORTED: {exc}\n")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary.get("status") == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())

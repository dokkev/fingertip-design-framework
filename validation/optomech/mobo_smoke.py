"""Run one complete production BO evaluation and verify resume."""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from lumo.optimization.ax_bo import run  # noqa: E402
from scripts import run_mobo as production  # noqa: E402


def _run(output_directory: Path) -> list[dict[str, object]]:
    return run(
        output_directory=output_directory,
        target_bo_trials=1,
        mechanics_preset=production.MECHANICS_PRESET,
        optical_preset=production.OPTICAL_PRESET,
        parameter_bounds_mm=production.PARAMETER_BOUNDS_MM,
        indenter_urdfs=production.INDENTER_URDFS,
        sphere_diameters_mm=production.SPHERE_DIAMETERS_MM,
        contact_y_mm=production.CONTACT_Y_MM,
        initial_clearance_m=production.INITIAL_CLEARANCE_M,
        force_targets_n=production.FORCE_TARGETS_N,
    )


def _completed_trial_indices(rows: list[dict[str, object]]) -> tuple[int, ...]:
    return tuple(
        int(row["ax_trial_index"]) for row in rows if row["status"] == "COMPLETED"
    )


def main() -> None:
    os.environ.setdefault("OTK_INCLUDE_DIR", str(production.OTK_INCLUDE_DIR))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output_directory = (
        _REPOSITORY_ROOT / "output" / "validation" / "mobo_smoke" / timestamp
    )
    print(
        f"Production BO smoke: one morphology evaluation; output={output_directory}",
        flush=True,
    )

    rows = _run(output_directory)
    completed_before_resume = _completed_trial_indices(rows)
    if len(completed_before_resume) != 1:
        raise RuntimeError("smoke did not produce exactly one completed BO morphology")

    resumed_rows = _run(output_directory)
    completed_after_resume = _completed_trial_indices(resumed_rows)
    if completed_after_resume != completed_before_resume:
        raise RuntimeError("resume lost or repeated the completed BO morphology")

    print("Production BO smoke: PASS", flush=True)
    print(f"completed Ax trial: {completed_after_resume[0]}", flush=True)
    print(f"outputs: {output_directory}", flush=True)


if __name__ == "__main__":
    main()

"""Index format-v3 contact datasets without repairing stored metadata."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from experiments.data_collection.contact_dataset import (
    DatasetFrameRecord,
    RunMetadata,
    SessionMetadata,
    iter_dataset_frames,
    parse_force_directory,
)


@dataclass(frozen=True)
class SessionIndex:
    """Resolved session data and explicit acquisition-coverage observations."""

    path: Path
    session: SessionMetadata
    runs: tuple[RunMetadata, ...]
    frames: tuple[DatasetFrameRecord, ...]
    coverage_rows: tuple[dict[str, Any], ...]
    issues: tuple[str, ...]
    unloaded_capture_count: int

    @property
    def specimen_id(self) -> str:
        return self.session.specimen_id


def index_session(
    session_path: str | Path,
    *,
    expected_repetitions: int = 5,
) -> SessionIndex:
    """Read one session and report missing, duplicate, and unusual coverage."""

    if expected_repetitions < 1:
        raise ValueError("expected_repetitions must be positive")
    root = Path(session_path).resolve()
    session = SessionMetadata.from_dict(_read_json(root / "session.json"))
    frames = tuple(iter_dataset_frames(root))
    issues: list[str] = []
    for frame in frames:
        if not frame.rgb_path.is_file():
            issues.append(f"missing image: {frame.rgb_path}")

    run_paths = sorted((root / "runs").glob("run_*"))
    runs_with_paths: list[tuple[Path, RunMetadata]] = []
    for path in run_paths:
        metadata_path = path / "run.json"
        if not path.is_dir() or not metadata_path.is_file():
            continue
        run = RunMetadata.from_dict(_read_json(metadata_path))
        if run.run_id != path.name:
            issues.append(
                f"run identity mismatch: directory {path.name}, metadata {run.run_id}"
            )
        runs_with_paths.append((path, run))

    identities: dict[tuple[str, int, int], list[tuple[Path, RunMetadata]]] = (
        defaultdict(list)
    )
    for path, run in runs_with_paths:
        identities[(run.indenter, run.hole_index, run.repetition_index)].append(
            (path, run)
        )
    for identity, matches in identities.items():
        if len(matches) > 1:
            issues.append(
                "duplicate condition identity "
                f"{identity}: {', '.join(run.run_id for _, run in matches)}"
            )

    frame_counts: Counter[tuple[str, float]] = Counter()
    for frame in frames:
        if frame.run is not None and frame.target_force_n is not None:
            frame_counts[(frame.run.run_id, frame.target_force_n)] += 1

    expected_forces = tuple(session.force_sequence.target_forces_n)
    expected_frame_count = session.force_sequence.expected_record_frame_count
    indenters = sorted({run.indenter for _, run in runs_with_paths})
    coverage: list[dict[str, Any]] = []
    expected_repetition_set = set(range(1, expected_repetitions + 1))
    for indenter in indenters:
        for hole in range(1, 7):
            for repetition in range(1, expected_repetitions + 1):
                matches = identities.get((indenter, hole, repetition), [])
                for target in expected_forces:
                    count = sum(
                        frame_counts[(run.run_id, target)] for _, run in matches
                    )
                    if not matches:
                        validity = "missing_run"
                    elif len(matches) > 1:
                        validity = "duplicate_run_identity"
                    elif matches[0][1].status != "complete":
                        validity = f"run_{matches[0][1].status}"
                    elif count == 0:
                        validity = "missing_force_segment"
                    elif count != expected_frame_count:
                        validity = "unexpected_frame_count"
                    else:
                        validity = "valid"
                    coverage.append(
                        _coverage_row(
                            session,
                            indenter,
                            hole,
                            repetition,
                            target,
                            matches,
                            count,
                            expected_frame_count,
                            validity,
                        )
                    )

    for (indenter, hole, repetition), matches in identities.items():
        if repetition in expected_repetition_set:
            continue
        for target in expected_forces:
            count = sum(frame_counts[(run.run_id, target)] for _, run in matches)
            coverage.append(
                _coverage_row(
                    session,
                    indenter,
                    hole,
                    repetition,
                    target,
                    matches,
                    count,
                    expected_frame_count,
                    "unexpected_repetition",
                )
            )
        issues.append(f"unexpected repetition {repetition} for {indenter}, hole {hole}")

    expected_force_set = set(expected_forces)
    for run_path, run in runs_with_paths:
        for force_path in sorted(run_path.glob("force_*N")):
            if not force_path.is_dir():
                continue
            try:
                target = parse_force_directory(force_path.name)
            except ValueError:
                issues.append(f"unexpected force directory: {force_path}")
                continue
            if target in expected_force_set:
                continue
            count = sum(1 for _ in (force_path / "frames").glob("*.png"))
            coverage.append(
                _coverage_row(
                    session,
                    run.indenter,
                    run.hole_index,
                    run.repetition_index,
                    target,
                    [(run_path, run)],
                    count,
                    expected_frame_count,
                    "unexpected_force",
                )
            )
            issues.append(f"unexpected force directory: {force_path}")

    unloaded_captures = {
        frame.segment_path.resolve() for frame in frames if frame.run is None
    }
    return SessionIndex(
        path=root,
        session=session,
        runs=tuple(run for _, run in runs_with_paths),
        frames=frames,
        coverage_rows=tuple(coverage),
        issues=tuple(sorted(set(issues))),
        unloaded_capture_count=len(unloaded_captures),
    )


def camera_consistency_warnings(indexes: list[SessionIndex]) -> list[str]:
    """Warn when absolute camera DN values are not directly comparable."""

    if len(indexes) < 2:
        return []
    fields = (
        "camera_model",
        "camera_serial_number",
        "camera_width",
        "camera_height",
        "camera_fps",
        "camera_exposure_us",
        "camera_gain",
        "camera_white_balance_k",
    )
    warnings = []
    for field in fields:
        values = {getattr(index.session, field) for index in indexes}
        if len(values) > 1:
            warnings.append(f"camera mismatch for {field}: {sorted(map(str, values))}")
    return warnings


def _coverage_row(
    session: SessionMetadata,
    indenter: str,
    hole: int,
    repetition: int,
    target: float,
    matches: list[tuple[Path, RunMetadata]],
    frame_count: int,
    expected_frame_count: int,
    validity: str,
) -> dict[str, Any]:
    return {
        "specimen_id": session.specimen_id,
        "material": session.material,
        "morphology": session.morphology,
        "indenter": indenter,
        "hole_index": hole,
        "repetition_index": repetition,
        "target_force_n": target,
        "run_ids": ";".join(run.run_id for _, run in matches),
        "run_statuses": ";".join(run.status for _, run in matches),
        "frame_count": frame_count,
        "expected_frame_count": expected_frame_count,
        "validity": validity,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


__all__ = ["SessionIndex", "camera_consistency_warnings", "index_session"]

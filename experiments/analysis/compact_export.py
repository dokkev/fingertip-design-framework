"""Export upload-sized HDF5 copies of physical RGB/force observations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from experiments.analysis.dataset import index_session
from experiments.analysis.optical import (
    PROFILE_BINS,
    OpticalStrip,
    calibrate_optical_strip,
    load_rgb,
    longitudinal_green_profile,
    temporal_median_rgb,
    warp_rgb,
)


FORMAT_NAME = "lumo_compact_physical_data"
FORMAT_VERSION = 1
MAP_HEIGHT = 128
MAP_WIDTH = 64
MAXIMUM_UPLOAD_BYTES = 500_000_000
_FRAME_CHUNK_COUNT = 16


@dataclass(frozen=True)
class CompactFrameSource:
    """One preserved camera observation and its directly associated metadata."""

    image_path: Path
    source_path: str
    session_index: int
    frame_index: int
    camera_host_time_s: float
    camera_device_timestamp_ms: float
    actual_force_n: float
    target_force_n: float
    wrench: tuple[float, float, float, float, float, float]
    unloaded: bool
    capture_id: str
    run_id: str
    indenter: str
    hole_index: int
    repetition_index: int
    contact_position_mm: float
    trajectory_elapsed_s: float
    cycle_index: int
    cycle_role: str
    phase: str


@dataclass(frozen=True)
class CompactSessionSource:
    """One source session and all observations exported from it."""

    path: Path
    name: str
    material: str
    morphology: str
    specimen_id: str
    frames: tuple[CompactFrameSource, ...]
    metadata_payload: bytes


@dataclass(frozen=True)
class CaptureCalibration:
    """One unloaded-capture geometry retained in the compact artifact."""

    session_index: int
    capture_id: str
    median_host_time_s: float
    strip: OpticalStrip
    reference_rgb: np.ndarray


def discover_session_directories(root: str | Path) -> list[Path]:
    """Return immediate child directories containing a session manifest."""

    directory = Path(root).resolve()
    sessions = sorted(
        path
        for path in directory.iterdir()
        if path.is_dir() and (path / "session.json").is_file()
    )
    if not sessions:
        raise RuntimeError(f"no session directories found under {directory}")
    return sessions


def export_compact_physical_data(
    contact_dataset_root: str | Path,
    contact_history_root: str | Path,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Export both physical acquisition schemas and verify the upload budget."""

    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    contact_path = output / "contact_dataset.h5"
    history_path = output / "contact_history.h5"
    for path in (contact_path, history_path):
        if path.exists() or path.with_suffix(path.suffix + ".partial").exists():
            raise FileExistsError(f"refusing to overwrite existing export: {path}")

    contact_sessions = [
        _index_contact_session(path, index)
        for index, path in enumerate(discover_session_directories(contact_dataset_root))
    ]
    history_sessions = [
        _index_history_session(path, index)
        for index, path in enumerate(discover_session_directories(contact_history_root))
    ]
    _export_hdf5(contact_path, "contact_dataset", contact_sessions)
    _export_hdf5(history_path, "contact_history", history_sessions)

    total_bytes = contact_path.stat().st_size + history_path.stat().st_size
    if total_bytes >= MAXIMUM_UPLOAD_BYTES:
        raise RuntimeError(
            "compact exports exceed the 500 MB upload target: "
            f"{total_bytes / 1_000_000:.3f} MB"
        )
    _write_readme(output, contact_path, history_path, contact_sessions, history_sessions)
    return contact_path, history_path


def verify_compact_hdf5(path: str | Path) -> dict[str, int | str]:
    """Check the structural and numerical invariants of one compact artifact."""

    artifact = Path(path).resolve()
    with h5py.File(artifact, "r") as data:
        if data.attrs["format_name"] != FORMAT_NAME:
            raise ValueError(f"unexpected compact format in {artifact}")
        if int(data.attrs["format_version"]) != FORMAT_VERSION:
            raise ValueError(f"unsupported compact format version in {artifact}")
        frame_count = int(data.attrs["frame_count"])
        session_count = int(data.attrs["session_count"])
        calibration_count = int(data.attrs["calibration_count"])
        dataset_kind = str(data.attrs["dataset_kind"])
        if data["frames/rgb"].shape != (frame_count, MAP_HEIGHT, MAP_WIDTH, 3):
            raise ValueError(f"invalid compact RGB shape in {artifact}")
        if data["frames/green_profile"].shape != (frame_count, PROFILE_BINS):
            raise ValueError(f"invalid Green-profile shape in {artifact}")
        if data["frames/session_index"].shape != (frame_count,):
            raise ValueError(f"invalid frame metadata length in {artifact}")
        if len(data["sessions/name"]) != session_count:
            raise ValueError(f"invalid session metadata length in {artifact}")
        if len(data["calibrations/capture_id"]) != calibration_count:
            raise ValueError(f"invalid calibration metadata length in {artifact}")
        if not np.all(np.isfinite(data["frames/green_profile"][:])):
            raise ValueError(f"non-finite Green profile in {artifact}")
        calibration_index = data["frames/calibration_index"][:]
        if np.any(calibration_index < 0) or np.any(
            calibration_index >= calibration_count
        ):
            raise ValueError(f"invalid frame calibration index in {artifact}")
    return {
        "dataset_kind": dataset_kind,
        "session_count": session_count,
        "frame_count": frame_count,
        "calibration_count": calibration_count,
    }


def _index_contact_session(path: Path, session_index: int) -> CompactSessionSource:
    index = index_session(path, expected_repetitions=5)
    if index.issues:
        raise RuntimeError(f"{path} has dataset-integrity issues: {index.issues}")
    invalid = [row for row in index.coverage_rows if row["validity"] != "valid"]
    if invalid:
        raise RuntimeError(f"{path} has {len(invalid)} invalid run-force states")

    frames: list[CompactFrameSource] = []
    for frame in index.frames:
        row = frame.measurements
        unloaded = frame.run is None
        actual_force = _force_magnitude(row)
        frames.append(
            CompactFrameSource(
                image_path=frame.rgb_path,
                source_path=f"{path.name}/{frame.rgb_path.relative_to(path)}",
                session_index=session_index,
                frame_index=int(row["frame_index"]),
                camera_host_time_s=float(row["camera_host_time_s"]),
                camera_device_timestamp_ms=float(row["camera_device_timestamp_ms"]),
                actual_force_n=actual_force,
                target_force_n=(
                    math.nan if frame.target_force_n is None else frame.target_force_n
                ),
                wrench=_wrench(row),
                unloaded=unloaded,
                capture_id=frame.segment_path.name if unloaded else "",
                run_id="" if unloaded else frame.run.run_id,
                indenter="" if unloaded else frame.run.indenter,
                hole_index=-1 if unloaded else frame.run.hole_index,
                repetition_index=-1 if unloaded else frame.run.repetition_index,
                contact_position_mm=math.nan,
                trajectory_elapsed_s=math.nan,
                cycle_index=-1,
                cycle_role="",
                phase="",
            )
        )
    return CompactSessionSource(
        path=path,
        name=path.name,
        material=index.session.material,
        morphology=index.session.morphology,
        specimen_id=index.session.specimen_id,
        frames=tuple(frames),
        metadata_payload=_metadata_payload(path),
    )


def _index_history_session(path: Path, session_index: int) -> CompactSessionSource:
    session_json = _read_json(path / "session.json")
    specimen = session_json["specimen"]
    frames: list[CompactFrameSource] = []
    for capture_path in sorted((path / "unloaded").glob("capture_*")):
        for row in _read_csv(capture_path / "frames.csv"):
            image_path = capture_path / row["rgb_filename"]
            frames.append(
                _history_frame(
                    image_path,
                    f"{path.name}/{image_path.relative_to(path)}",
                    session_index,
                    row,
                    unloaded=True,
                    capture_id=capture_path.name,
                )
            )

    for run_path in sorted((path / "runs").glob("run_*")):
        if not run_path.is_dir():
            continue
        run = _read_json(run_path / "run.json")
        if run["run_id"] != run_path.name:
            raise RuntimeError(
                f"run identity mismatch: {run_path.name}, {run['run_id']}"
            )
        if run["status"] != "complete":
            raise RuntimeError(f"refusing incomplete contact-history run: {run_path}")
        trajectory_path = run_path / "trajectory"
        for row in _read_csv(trajectory_path / "frames.csv"):
            image_path = trajectory_path / row["rgb_filename"]
            frames.append(
                _history_frame(
                    image_path,
                    f"{path.name}/{image_path.relative_to(path)}",
                    session_index,
                    row,
                    unloaded=False,
                    run=run,
                )
            )
    if not frames or not any(frame.unloaded for frame in frames):
        raise RuntimeError(f"{path} has no exportable frames or unloaded capture")
    return CompactSessionSource(
        path=path,
        name=path.name,
        material=str(specimen["material"]),
        morphology=str(specimen["morphology"]),
        specimen_id=str(specimen["specimen_id"]),
        frames=tuple(frames),
        metadata_payload=_metadata_payload(path),
    )


def _history_frame(
    image_path: Path,
    source_path: str,
    session_index: int,
    row: dict[str, str],
    *,
    unloaded: bool,
    capture_id: str = "",
    run: dict[str, Any] | None = None,
) -> CompactFrameSource:
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    return CompactFrameSource(
        image_path=image_path,
        source_path=source_path,
        session_index=session_index,
        frame_index=int(row["frame_index"]),
        camera_host_time_s=float(row["camera_host_time_s"]),
        camera_device_timestamp_ms=float(row["camera_device_timestamp_ms"]),
        actual_force_n=float(row.get("force_magnitude_N") or _force_magnitude(row)),
        target_force_n=(
            math.nan if unloaded else float(row["target_force_N"])
        ),
        wrench=_wrench(row),
        unloaded=unloaded,
        capture_id=capture_id,
        run_id="" if run is None else str(run["run_id"]),
        indenter="" if run is None else str(run["indenter"]),
        hole_index=-1 if run is None else int(run["hole_index"]),
        repetition_index=-1 if run is None else int(run["repetition_index"]),
        contact_position_mm=(
            math.nan if run is None else float(run["contact_position_mm"])
        ),
        trajectory_elapsed_s=(
            math.nan if unloaded else float(row["trajectory_elapsed_s"])
        ),
        cycle_index=-1 if unloaded else int(row["cycle_index"]),
        cycle_role="" if unloaded else row["cycle_role"],
        phase="" if unloaded else row["phase"],
    )


def _export_hdf5(
    output_path: Path,
    dataset_kind: str,
    sessions: list[CompactSessionSource],
) -> None:
    frames = [frame for session in sessions for frame in session.frames]
    calibrations = _calibrate_captures(sessions)
    calibration_indices = _assign_calibrations(frames, calibrations)
    partial = output_path.with_suffix(output_path.suffix + ".partial")

    with h5py.File(partial, "w") as data:
        data.attrs["format_name"] = FORMAT_NAME
        data.attrs["format_version"] = FORMAT_VERSION
        data.attrs["dataset_kind"] = dataset_kind
        data.attrs["frame_count"] = len(frames)
        data.attrs["session_count"] = len(sessions)
        data.attrs["calibration_count"] = len(calibrations)
        data.attrs["map_height"] = MAP_HEIGHT
        data.attrs["map_width"] = MAP_WIDTH
        data.attrs["profile_bins"] = PROFILE_BINS
        data.attrs["rgb_semantics"] = (
            "absolute RGB8 camera DN after unloaded-capture geometry remap and "
            "2x area downsampling; no unloaded subtraction or normalization"
        )

        _write_sessions(data.create_group("sessions"), sessions)
        _write_calibrations(data.create_group("calibrations"), calibrations)
        frame_group = data.create_group("frames")
        _write_frame_metadata(frame_group, frames, calibration_indices)
        rgb = frame_group.create_dataset(
            "rgb",
            shape=(len(frames), MAP_HEIGHT, MAP_WIDTH, 3),
            dtype=np.uint8,
            chunks=(min(_FRAME_CHUNK_COUNT, len(frames)), MAP_HEIGHT, MAP_WIDTH, 3),
            compression="gzip",
            compression_opts=6,
            shuffle=True,
            fletcher32=True,
        )
        profiles = frame_group.create_dataset(
            "green_profile",
            shape=(len(frames), PROFILE_BINS),
            dtype=np.float64,
            chunks=(min(256, len(frames)), PROFILE_BINS),
            compression="gzip",
            compression_opts=6,
            shuffle=True,
            fletcher32=True,
        )

        for start in range(0, len(frames), _FRAME_CHUNK_COUNT):
            stop = min(start + _FRAME_CHUNK_COUNT, len(frames))
            maps: list[np.ndarray] = []
            chunk_profiles: list[np.ndarray] = []
            for frame, calibration_index in zip(
                frames[start:stop], calibration_indices[start:stop], strict=True
            ):
                image = load_rgb(frame.image_path)
                strip = calibrations[int(calibration_index)].strip
                canonical = warp_rgb(image, strip)
                maps.append(
                    cv2.resize(
                        canonical,
                        (MAP_WIDTH, MAP_HEIGHT),
                        interpolation=cv2.INTER_AREA,
                    )
                )
                profile, _ = longitudinal_green_profile(
                    image, strip, bins=PROFILE_BINS
                )
                chunk_profiles.append(profile)
            rgb[start:stop] = np.asarray(maps, dtype=np.uint8)
            profiles[start:stop] = np.asarray(chunk_profiles, dtype=np.float64)
            if stop == len(frames) or stop % 512 < _FRAME_CHUNK_COUNT:
                print(f"{dataset_kind}: {stop}/{len(frames)} frames")

    partial.replace(output_path)
    verify_compact_hdf5(output_path)


def _calibrate_captures(
    sessions: list[CompactSessionSource],
) -> list[CaptureCalibration]:
    calibrations: list[CaptureCalibration] = []
    for session in sessions:
        captures: dict[str, list[CompactFrameSource]] = {}
        for frame in session.frames:
            if frame.unloaded:
                captures.setdefault(frame.capture_id, []).append(frame)
        if not captures:
            raise RuntimeError(f"{session.path} has no unloaded capture")
        for capture_id, frames in sorted(captures.items()):
            images = [load_rgb(frame.image_path) for frame in frames]
            reference = temporal_median_rgb(images)
            try:
                strip = calibrate_optical_strip(reference)
            except RuntimeError as error:
                raise RuntimeError(
                    f"{session.name}/{capture_id} calibration failed: {error}"
                ) from error
            calibrations.append(
                CaptureCalibration(
                    session_index=frames[0].session_index,
                    capture_id=capture_id,
                    median_host_time_s=float(
                        np.median([frame.camera_host_time_s for frame in frames])
                    ),
                    strip=strip,
                    reference_rgb=reference,
                )
            )
    return calibrations


def _assign_calibrations(
    frames: list[CompactFrameSource],
    calibrations: list[CaptureCalibration],
) -> np.ndarray:
    by_session: dict[int, list[int]] = {}
    exact: dict[tuple[int, str], int] = {}
    for index, calibration in enumerate(calibrations):
        by_session.setdefault(calibration.session_index, []).append(index)
        exact[(calibration.session_index, calibration.capture_id)] = index

    result = np.empty(len(frames), dtype=np.int16)
    for index, frame in enumerate(frames):
        if frame.unloaded:
            result[index] = exact[(frame.session_index, frame.capture_id)]
            continue
        candidates = by_session[frame.session_index]
        result[index] = min(
            candidates,
            key=lambda candidate: abs(
                calibrations[candidate].median_host_time_s - frame.camera_host_time_s
            ),
        )
    return result


def _write_sessions(group: h5py.Group, sessions: list[CompactSessionSource]) -> None:
    _write_strings(group, "name", [session.name for session in sessions])
    _write_strings(group, "material", [session.material for session in sessions])
    _write_strings(group, "morphology", [session.morphology for session in sessions])
    _write_strings(group, "specimen_id", [session.specimen_id for session in sessions])
    _write_strings(group, "source_path", [str(session.path) for session in sessions])
    metadata = group.create_group("source_metadata")
    for index, session in enumerate(sessions):
        metadata.create_dataset(
            f"session_{index:03d}",
            data=np.frombuffer(session.metadata_payload, dtype=np.uint8),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            fletcher32=True,
        )


def _write_calibrations(
    group: h5py.Group, calibrations: list[CaptureCalibration]
) -> None:
    _write_numeric(
        group,
        "session_index",
        np.asarray([item.session_index for item in calibrations], dtype=np.int16),
    )
    _write_strings(group, "capture_id", [item.capture_id for item in calibrations])
    _write_numeric(
        group,
        "median_host_time_s",
        np.asarray([item.median_host_time_s for item in calibrations], dtype=np.float64),
    )
    _write_numeric(
        group,
        "source_mask",
        np.asarray([item.strip.source_mask for item in calibrations], dtype=np.uint8),
    )
    _write_numeric(
        group,
        "map_x",
        np.asarray([item.strip.map_x for item in calibrations], dtype=np.float32),
    )
    _write_numeric(
        group,
        "map_y",
        np.asarray([item.strip.map_y for item in calibrations], dtype=np.float32),
    )
    _write_numeric(
        group,
        "support_mask_full",
        np.asarray([item.strip.support_mask for item in calibrations], dtype=np.uint8),
    )
    compact_support = [
        cv2.resize(
            item.strip.support_mask.astype(np.uint8),
            (MAP_WIDTH, MAP_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        for item in calibrations
    ]
    _write_numeric(
        group,
        "support_mask",
        np.asarray(compact_support, dtype=np.uint8),
    )
    compact_reference = [
        cv2.resize(
            warp_rgb(item.reference_rgb, item.strip),
            (MAP_WIDTH, MAP_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        for item in calibrations
    ]
    _write_numeric(
        group,
        "reference_rgb",
        np.asarray(compact_reference, dtype=np.uint8),
    )


def _write_frame_metadata(
    group: h5py.Group,
    frames: list[CompactFrameSource],
    calibration_indices: np.ndarray,
) -> None:
    _write_strings(group, "source_path", [frame.source_path for frame in frames])
    _write_numeric(
        group,
        "session_index",
        np.asarray([frame.session_index for frame in frames], dtype=np.int16),
    )
    _write_numeric(group, "calibration_index", calibration_indices)
    _write_numeric(
        group,
        "frame_index",
        np.asarray([frame.frame_index for frame in frames], dtype=np.int32),
    )
    _write_numeric(
        group,
        "camera_host_time_s",
        np.asarray([frame.camera_host_time_s for frame in frames], dtype=np.float64),
    )
    _write_numeric(
        group,
        "camera_device_timestamp_ms",
        np.asarray(
            [frame.camera_device_timestamp_ms for frame in frames], dtype=np.float64
        ),
    )
    _write_numeric(
        group,
        "actual_force_n",
        np.asarray([frame.actual_force_n for frame in frames], dtype=np.float64),
    )
    _write_numeric(
        group,
        "target_force_n",
        np.asarray([frame.target_force_n for frame in frames], dtype=np.float64),
    )
    _write_numeric(
        group,
        "wrench",
        np.asarray([frame.wrench for frame in frames], dtype=np.float64),
    )
    _write_numeric(
        group,
        "unloaded",
        np.asarray([frame.unloaded for frame in frames], dtype=np.uint8),
    )
    _write_strings(group, "capture_id", [frame.capture_id for frame in frames])
    _write_strings(group, "run_id", [frame.run_id for frame in frames])
    _write_strings(group, "indenter", [frame.indenter for frame in frames])
    _write_numeric(
        group,
        "hole_index",
        np.asarray([frame.hole_index for frame in frames], dtype=np.int16),
    )
    _write_numeric(
        group,
        "repetition_index",
        np.asarray([frame.repetition_index for frame in frames], dtype=np.int16),
    )
    _write_numeric(
        group,
        "contact_position_mm",
        np.asarray([frame.contact_position_mm for frame in frames], dtype=np.float64),
    )
    _write_numeric(
        group,
        "trajectory_elapsed_s",
        np.asarray([frame.trajectory_elapsed_s for frame in frames], dtype=np.float64),
    )
    _write_numeric(
        group,
        "cycle_index",
        np.asarray([frame.cycle_index for frame in frames], dtype=np.int16),
    )
    _write_strings(group, "cycle_role", [frame.cycle_role for frame in frames])
    _write_strings(group, "phase", [frame.phase for frame in frames])


def _write_numeric(group: h5py.Group, name: str, values: np.ndarray) -> None:
    group.create_dataset(
        name,
        data=values,
        compression="gzip",
        compression_opts=6,
        shuffle=True,
        fletcher32=True,
    )


def _write_strings(group: h5py.Group, name: str, values: list[str]) -> None:
    encoded = [value.encode("utf-8") for value in values]
    width = max(1, max(map(len, encoded), default=1))
    group.create_dataset(
        name,
        data=np.asarray(encoded, dtype=f"S{width}"),
        compression="gzip",
        compression_opts=6,
        shuffle=True,
        fletcher32=True,
    )


def _metadata_payload(session_path: Path) -> bytes:
    files = {
        str(path.relative_to(session_path)): path.read_text(encoding="utf-8")
        for path in sorted(session_path.rglob("*"))
        if path.is_file() and path.suffix in {".csv", ".json"}
    }
    return json.dumps(
        {"source_session": session_path.name, "text_files": files},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_readme(
    output: Path,
    contact_path: Path,
    history_path: Path,
    contact_sessions: list[CompactSessionSource],
    history_sessions: list[CompactSessionSource],
) -> None:
    contact = verify_compact_hdf5(contact_path)
    history = verify_compact_hdf5(history_path)
    total_bytes = contact_path.stat().st_size + history_path.stat().st_size
    text = f"""# LUMO compact physical data

These HDF5 files are upload-sized scientific copies of the final physical
contact datasets. They are not full-resolution camera archives.

| file | sessions | observations | size |
|---|---:|---:|---:|
| `contact_dataset.h5` | {len(contact_sessions)} | {contact['frame_count']} | {contact_path.stat().st_size / 1_000_000:.3f} MB |
| `contact_history.h5` | {len(history_sessions)} | {history['frame_count']} | {history_path.stat().st_size / 1_000_000:.3f} MB |
| **total** | {len(contact_sessions) + len(history_sessions)} | {int(contact['frame_count']) + int(history['frame_count'])} | **{total_bytes / 1_000_000:.3f} MB** |

Every source observation is retained individually. `frames/rgb` stores absolute
camera RGB8 DN on a 128 x 64 unloaded-capture-derived canonical map. It applies
no unloaded subtraction, intensity normalization, or response transformation.
`frames/green_profile` stores the 128-bin profile calculated from the full
256 x 128 canonical strip before map downsampling.

`frames/calibration_index` identifies the independently preserved unloaded
capture geometry used for each map. Unloaded frames use their own capture;
loaded frames use the temporally nearest capture from the same session for
geometry only. All unloaded captures remain separate in `calibrations/`.

The typed frame arrays retain force, wrench, timestamps, run identity, indenter,
hole, repetition, and history-cycle fields. `sessions/source_metadata/` embeds
the complete source JSON/CSV text, including session, run, trajectory, and
provenance manifests.

What is intentionally omitted: full 1920 x 1080 PNG pixels and scene content
outside the calibrated fingertip strip. Keep the original PNG directories as
the authoritative archival data when full-frame segmentation, camera-pose, or
mechanical image-tracking methods must be rerun.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _force_magnitude(row: dict[str, str]) -> float:
    return float(
        math.sqrt(sum(float(row[name]) ** 2 for name in ("Fx_N", "Fy_N", "Fz_N")))
    )


def _wrench(row: dict[str, str]) -> tuple[float, float, float, float, float, float]:
    return tuple(
        float(row[name])
        for name in ("Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm")
    )  # type: ignore[return-value]


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "MAP_HEIGHT",
    "MAP_WIDTH",
    "discover_session_directories",
    "export_compact_physical_data",
    "verify_compact_hdf5",
]

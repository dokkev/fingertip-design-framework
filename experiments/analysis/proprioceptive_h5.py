"""Compact one proprioceptive-force run into an upload-bounded HDF5 artifact."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np


FORMAT_NAME = "lumo_proprioceptive_force_run"
DATASET_FORMAT_NAME = "lumo_proprioceptive_force_dataset"
FORMAT_VERSION = 1
MAXIMUM_UPLOAD_BYTES = 500_000_000
_BYTE_CHUNK_SIZE = 1_048_576
_SOURCE_FILENAMES = (
    "metadata.json",
    "motor.csv",
    "ft.csv",
    "optical.csv",
    "force_estimate.csv",
    "force_sequence.csv",
    "camera_timestamps.csv",
)


@dataclass(frozen=True)
class ProprioceptiveH5Summary:
    """Verified compact-artifact identity and size."""

    path: Path
    frame_count: int
    unloaded_reference_count: int
    contact_frame_count: int
    jpeg_quality: int
    size_bytes: int


@dataclass(frozen=True)
class ProprioceptiveDatasetH5Summary:
    """Verified multi-run compact-artifact identity and size."""

    path: Path
    run_count: int
    frame_count: int
    unloaded_reference_count: int
    contact_frame_count: int
    jpeg_quality: int
    size_bytes: int


def export_proprioceptive_run_h5(
    run_path: str | Path,
    output_path: str | Path,
    *,
    jpeg_quality: int = 95,
    maximum_bytes: int = MAXIMUM_UPLOAD_BYTES,
) -> ProprioceptiveH5Summary:
    """Preserve one run as full-resolution JPEG frames plus exact source tables."""

    run = Path(run_path).resolve()
    output = Path(output_path).resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    if not run.is_dir():
        raise NotADirectoryError(f"run directory does not exist: {run}")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    if maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite existing export: {output}")

    rows = _read_camera_rows(run)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_hdf5(run, rows, partial, jpeg_quality)
        size_bytes = partial.stat().st_size
        if size_bytes >= maximum_bytes:
            raise RuntimeError(
                f"compact run is {size_bytes / 1_000_000:.3f} MB, exceeding "
                f"the {maximum_bytes / 1_000_000:.3f} MB limit; reduce the "
                "recording duration/rate or explicitly choose a lower JPEG quality"
            )
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return verify_proprioceptive_h5(output, maximum_bytes=maximum_bytes)


def verify_proprioceptive_h5(
    path: str | Path,
    *,
    maximum_bytes: int = MAXIMUM_UPLOAD_BYTES,
) -> ProprioceptiveH5Summary:
    """Verify schema, frame index, decodability, and upload-size limit."""

    artifact = Path(path).resolve()
    size_bytes = artifact.stat().st_size
    if size_bytes >= maximum_bytes:
        raise RuntimeError(
            f"artifact is {size_bytes / 1_000_000:.3f} MB, exceeding the "
            f"{maximum_bytes / 1_000_000:.3f} MB limit"
        )
    with h5py.File(artifact, "r") as data:
        if str(data.attrs["format_name"]) != FORMAT_NAME:
            raise ValueError(f"unexpected format_name in {artifact}")
        if int(data.attrs["format_version"]) != FORMAT_VERSION:
            raise ValueError(f"unsupported format_version in {artifact}")
        frame_count, unloaded, quality = _verify_run_group(data, str(artifact))
    return ProprioceptiveH5Summary(
        path=artifact,
        frame_count=frame_count,
        unloaded_reference_count=unloaded,
        contact_frame_count=frame_count - unloaded,
        jpeg_quality=quality,
        size_bytes=size_bytes,
    )


def export_proprioceptive_dataset_h5(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    jpeg_quality: int = 95,
    maximum_bytes: int = MAXIMUM_UPLOAD_BYTES,
) -> ProprioceptiveDatasetH5Summary:
    """Preserve every run in one upload-bounded HDF5 artifact."""

    dataset = Path(dataset_path).resolve()
    output = Path(output_path).resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    if not dataset.is_dir():
        raise NotADirectoryError(f"dataset directory does not exist: {dataset}")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    if maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite existing export: {output}")

    runs = sorted(
        path
        for path in dataset.iterdir()
        if path.is_dir() and (path / "camera_timestamps.csv").is_file()
    )
    if not runs:
        raise RuntimeError(f"dataset contains no completed run directories: {dataset}")
    rows_by_run = [(run, _read_camera_rows(run)) for run in runs]

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(partial, "w") as data:
            data.attrs["format_name"] = DATASET_FORMAT_NAME
            data.attrs["format_version"] = FORMAT_VERSION
            data.attrs["source_dataset"] = dataset.name
            data.attrs["run_count"] = len(runs)
            data.attrs["jpeg_quality"] = jpeg_quality
            data.attrs["image_codec"] = "jpeg"
            data.attrs["image_semantics"] = (
                "full-resolution camera RGB8 encoded independently as "
                "high-quality JPEG; no crop, resize, subtraction, or normalization"
            )
            _write_strings(data, "run_ids", [run.name for run in runs])
            run_groups = data.create_group("runs")
            for run, rows in rows_by_run:
                _write_run_group(
                    run_groups.create_group(run.name),
                    run,
                    rows,
                    jpeg_quality,
                )
        size_bytes = partial.stat().st_size
        if size_bytes >= maximum_bytes:
            raise RuntimeError(
                f"compact dataset is {size_bytes / 1_000_000:.3f} MB, exceeding "
                f"the {maximum_bytes / 1_000_000:.3f} MB limit; choose a lower "
                "JPEG quality"
            )
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return verify_proprioceptive_dataset_h5(output, maximum_bytes=maximum_bytes)


def verify_proprioceptive_dataset_h5(
    path: str | Path,
    *,
    maximum_bytes: int = MAXIMUM_UPLOAD_BYTES,
) -> ProprioceptiveDatasetH5Summary:
    """Verify every run index and sampled JPEG in a multi-run artifact."""

    artifact = Path(path).resolve()
    size_bytes = artifact.stat().st_size
    if size_bytes >= maximum_bytes:
        raise RuntimeError(
            f"artifact is {size_bytes / 1_000_000:.3f} MB, exceeding the "
            f"{maximum_bytes / 1_000_000:.3f} MB limit"
        )
    with h5py.File(artifact, "r") as data:
        if str(data.attrs["format_name"]) != DATASET_FORMAT_NAME:
            raise ValueError(f"unexpected format_name in {artifact}")
        if int(data.attrs["format_version"]) != FORMAT_VERSION:
            raise ValueError(f"unsupported format_version in {artifact}")
        quality = int(data.attrs["jpeg_quality"])
        run_ids = _decode_strings(data["run_ids"][:])
        if len(run_ids) != int(data.attrs["run_count"]):
            raise ValueError("run_ids has the wrong length")
        if run_ids != sorted(run_ids) or len(set(run_ids)) != len(run_ids):
            raise ValueError("run_ids must be sorted and unique")
        if set(data["runs"]) != set(run_ids):
            raise ValueError("runs group does not match run_ids")

        frame_count = 0
        unloaded_count = 0
        for run_id in run_ids:
            run_frames, run_unloaded, run_quality = _verify_run_group(
                data[f"runs/{run_id}"], f"{artifact}:{run_id}"
            )
            if run_quality != quality:
                raise ValueError(f"{run_id} JPEG quality differs from dataset")
            frame_count += run_frames
            unloaded_count += run_unloaded
    return ProprioceptiveDatasetH5Summary(
        path=artifact,
        run_count=len(run_ids),
        frame_count=frame_count,
        unloaded_reference_count=unloaded_count,
        contact_frame_count=frame_count - unloaded_count,
        jpeg_quality=quality,
        size_bytes=size_bytes,
    )


def _read_camera_rows(run: Path) -> list[dict[str, str]]:
    path = run / "camera_timestamps.csv"
    if not path.is_file():
        raise FileNotFoundError(f"camera index does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"camera index is empty: {path}")
    required = {
        "frame_index",
        "timestamp_ns",
        "filename",
        "capture_kind",
        "camera_device_timestamp_ms",
        "camera_frame_number",
        "ft_contact_force_N",
        "ft_timestamp_ns",
        "camera_ft_time_delta_ms",
    }
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"camera index is missing columns: {sorted(missing)}")
    for expected, row in enumerate(rows):
        if int(row["frame_index"]) != expected:
            raise RuntimeError("camera frame_index must be contiguous and zero-based")
        if not (run / row["filename"]).is_file():
            raise FileNotFoundError(
                f"camera frame does not exist: {run / row['filename']}"
            )
    return rows


def _write_hdf5(
    run: Path,
    rows: list[dict[str, str]],
    partial: Path,
    jpeg_quality: int,
) -> None:
    with h5py.File(partial, "w") as data:
        data.attrs["format_name"] = FORMAT_NAME
        data.attrs["format_version"] = FORMAT_VERSION
        _write_run_group(data, run, rows, jpeg_quality)


def _write_run_group(
    data: h5py.Group,
    run: Path,
    rows: list[dict[str, str]],
    jpeg_quality: int,
) -> None:
    """Write one run below either a file root or dataset run group."""

    data.attrs["source_run"] = run.name
    data.attrs["frame_count"] = len(rows)
    data.attrs["image_codec"] = "jpeg"
    data.attrs["jpeg_quality"] = jpeg_quality
    data.attrs["image_semantics"] = (
        "full-resolution camera RGB8 encoded independently as high-quality JPEG; "
        "no crop, resize, subtraction, or normalization"
    )

    camera = data.create_group("camera")
    _write_numeric(camera, "frame_index", rows, int, np.int64)
    _write_numeric(camera, "timestamp_ns", rows, int, np.int64)
    _write_strings(camera, "filename", [row["filename"] for row in rows])
    _write_strings(camera, "capture_kind", [row["capture_kind"] for row in rows])
    _write_numeric(camera, "camera_device_timestamp_ms", rows, float, np.float64)
    _write_numeric(camera, "camera_frame_number", rows, int, np.int64)
    _write_numeric(camera, "ft_contact_force_N", rows, float, np.float64)
    _write_numeric(camera, "ft_timestamp_ns", rows, int, np.int64)
    _write_numeric(camera, "camera_ft_time_delta_ms", rows, float, np.float64)

    offsets = camera.create_dataset("jpeg_offsets", (len(rows) + 1,), dtype=np.uint64)
    encoded_bytes = camera.create_dataset(
        "jpeg_bytes",
        shape=(0,),
        maxshape=(None,),
        dtype=np.uint8,
        chunks=(_BYTE_CHUNK_SIZE,),
    )
    offsets[0] = 0
    byte_offset = 0
    shape: tuple[int, int, int] | None = None
    for index, row in enumerate(rows):
        bgr = cv2.imread(str(run / row["filename"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read camera frame: {run / row['filename']}")
        if shape is None:
            shape = bgr.shape
            data.attrs["image_height"] = shape[0]
            data.attrs["image_width"] = shape[1]
            data.attrs["image_channels"] = shape[2]
        elif bgr.shape != shape:
            raise RuntimeError(
                f"camera frame shape changed from {shape} to {bgr.shape}"
            )
        success, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                jpeg_quality,
                cv2.IMWRITE_JPEG_OPTIMIZE,
                1,
            ],
        )
        if not success:
            raise RuntimeError(f"failed to encode camera frame {index}")
        next_offset = byte_offset + len(encoded)
        encoded_bytes.resize((next_offset,))
        encoded_bytes[byte_offset:next_offset] = encoded
        offsets[index + 1] = next_offset
        byte_offset = next_offset

    source = data.create_group("source_files")
    for filename in _SOURCE_FILENAMES:
        path = run / filename
        if not path.is_file():
            continue
        values = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        source.create_dataset(
            filename.replace(".", "_"),
            data=values,
            compression="gzip",
            compression_opts=6,
            shuffle=True,
            fletcher32=True,
        ).attrs["filename"] = filename


def _verify_run_group(data: h5py.Group, artifact_label: str) -> tuple[int, int, int]:
    frame_count = int(data.attrs["frame_count"])
    quality = int(data.attrs["jpeg_quality"])
    offsets = np.asarray(data["camera/jpeg_offsets"], dtype=np.uint64)
    byte_count = len(data["camera/jpeg_bytes"])
    if offsets.shape != (frame_count + 1,):
        raise ValueError(f"{artifact_label}: camera/jpeg_offsets has wrong length")
    if offsets[0] != 0 or offsets[-1] != byte_count or np.any(np.diff(offsets) <= 0):
        raise ValueError(f"{artifact_label}: camera JPEG byte offsets are invalid")
    for name in (
        "frame_index",
        "timestamp_ns",
        "capture_kind",
        "camera_device_timestamp_ms",
        "camera_frame_number",
        "ft_contact_force_N",
        "ft_timestamp_ns",
        "camera_ft_time_delta_ms",
    ):
        if len(data[f"camera/{name}"]) != frame_count:
            raise ValueError(f"{artifact_label}: camera/{name} has wrong length")
    capture_kind = _decode_strings(data["camera/capture_kind"][:])
    if any(kind not in {"unloaded_reference", "contact"} for kind in capture_kind):
        raise ValueError(f"{artifact_label}: invalid capture_kind")
    for index in sorted({0, frame_count - 1}):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        encoded = np.asarray(data["camera/jpeg_bytes"][start:stop], dtype=np.uint8)
        if cv2.imdecode(encoded, cv2.IMREAD_COLOR) is None:
            raise ValueError(f"{artifact_label}: frame {index} is not decodable")
    unloaded = sum(kind == "unloaded_reference" for kind in capture_kind)
    return frame_count, unloaded, quality


def _write_numeric(
    group: h5py.Group,
    name: str,
    rows: list[dict[str, str]],
    parser,
    dtype,
) -> None:
    values = np.asarray([parser(row[name]) for row in rows], dtype=dtype)
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


def _decode_strings(values: np.ndarray) -> list[str]:
    return [bytes(value).decode("utf-8") for value in values]


__all__ = [
    "DATASET_FORMAT_NAME",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "MAXIMUM_UPLOAD_BYTES",
    "ProprioceptiveH5Summary",
    "ProprioceptiveDatasetH5Summary",
    "export_proprioceptive_dataset_h5",
    "export_proprioceptive_run_h5",
    "verify_proprioceptive_dataset_h5",
    "verify_proprioceptive_h5",
]

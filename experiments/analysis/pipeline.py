"""Raw-image extraction and compact-cache orchestration."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .aggregation import aggregate_analysis
from .dataset_index import SessionIndex, camera_consistency_warnings, index_session
from .deformation import build_contour_reference, contour_deformation
from .export import export_analysis_bundle
from .optical_response import (
    DEFAULT_GREEN_EXCESS_THRESHOLD_DN,
    STORED_SIGNATURE_BINS,
    actual_force_magnitude,
    calibrate_analysis_strip,
    longitudinal_signature,
    optical_metrics,
    unloaded_median_rgb,
    warp_rgb,
)


_CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class AnalysisConfig:
    expected_repetitions: int = 5
    signature_bins: int = STORED_SIGNATURE_BINS
    green_excess_threshold_dn: float = DEFAULT_GREEN_EXCESS_THRESHOLD_DN
    deformation_contour_samples: int = 384
    deformation_search_radius_px: int = 12
    hole_spacing_mm: float | None = None


def analyze_sessions(
    session_paths: list[str | Path],
    output_path: str | Path,
    *,
    recompute: bool = False,
    config: AnalysisConfig = AnalysisConfig(),
) -> Path:
    """Analyze N sessions, export a compact bundle, and return its directory."""

    if not session_paths:
        raise ValueError("at least one session path is required")
    indexes = [
        index_session(path, expected_repetitions=config.expected_repetitions)
        for path in session_paths
    ]
    specimen_ids = [index.session_id for index in indexes]
    if len(specimen_ids) != len(set(specimen_ids)):
        raise ValueError("input sessions must have unique specimen_id values")
    output = Path(output_path).resolve()
    cache_root = output / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    frame_rows: list[dict[str, Any]] = []
    signatures: list[np.ndarray] = []
    unloaded_rows: list[dict[str, Any]] = []
    extraction_details: list[dict[str, Any]] = []
    for index in indexes:
        cache = cache_root / _cache_name(index)
        extracted = _extract_or_load(index, cache, config, recompute=recompute)
        frame_rows.extend(extracted["frame_rows"])
        signatures.extend(extracted["signatures"])
        unloaded_rows.extend(extracted["unloaded_rows"])
        extraction_details.append(extracted["details"])

    signature_array = np.asarray(signatures, dtype=np.float64)
    aggregated = aggregate_analysis(
        frame_rows,
        signature_array,
        hole_spacing_mm=config.hole_spacing_mm,
    )
    warnings = camera_consistency_warnings(indexes)
    return export_analysis_bundle(
        output,
        indexes=indexes,
        frame_rows=frame_rows,
        frame_signatures=signature_array,
        unloaded_rows=unloaded_rows,
        aggregated=aggregated,
        camera_warnings=warnings,
        extraction_details=extraction_details,
        config=asdict(config),
    )


def _extract_or_load(
    index: SessionIndex,
    cache: Path,
    config: AnalysisConfig,
    *,
    recompute: bool,
) -> dict[str, Any]:
    metadata_path = cache / "metadata.json"
    frame_path = cache / "frame_features.csv"
    signatures_path = cache / "frame_signatures.npz"
    unloaded_path = cache / "unloaded_stability.csv"
    if not recompute and all(
        path.is_file()
        for path in (metadata_path, frame_path, signatures_path, unloaded_path)
    ):
        metadata = _read_json(metadata_path)
        if (
            metadata.get("cache_format_version") == _CACHE_FORMAT_VERSION
            and metadata.get("source_session_path") == str(index.path)
            and metadata.get("analysis_config") == asdict(config)
        ):
            with np.load(signatures_path, allow_pickle=False) as arrays:
                signatures = arrays["signatures"].astype(np.float64)
            rows = _read_csv(frame_path)
            if len(rows) != len(signatures):
                raise RuntimeError(f"cache row/signature count mismatch: {cache}")
            return {
                "frame_rows": rows,
                "signatures": list(signatures),
                "unloaded_rows": _read_csv(unloaded_path),
                "details": {**metadata, "cache_reused": True},
            }

    cache.mkdir(parents=True, exist_ok=True)
    extracted = _extract_session(index, config)
    _write_csv(frame_path, extracted["frame_rows"])
    _write_csv(unloaded_path, extracted["unloaded_rows"])
    np.savez_compressed(
        signatures_path,
        signatures=np.asarray(extracted["signatures"], dtype=np.float32),
    )
    details = {
        "cache_format_version": _CACHE_FORMAT_VERSION,
        "source_session_path": str(index.path),
        "specimen_id": index.session_id,
        "analysis_config": asdict(config),
        "loaded_frame_count": len(extracted["frame_rows"]),
        "unloaded_frame_count": len(extracted["unloaded_rows"]),
        "calibration_mask_area_px": extracted["calibration_mask_area_px"],
        "unloaded_capture_geometry_centroid_span_px": extracted[
            "unloaded_capture_geometry_centroid_span_px"
        ],
        "cache_reused": False,
    }
    metadata_path.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    extracted["details"] = details
    return extracted


def _extract_session(index: SessionIndex, config: AnalysisConfig) -> dict[str, Any]:
    unloaded_records = [frame for frame in index.frames if frame.run is None]
    loaded_records = [frame for frame in index.frames if frame.run is not None]
    unloaded_images = [_load_rgb(frame.rgb_path) for frame in unloaded_records]
    reference = unloaded_median_rgb(unloaded_images)
    capture_centroids = []
    for capture_path in sorted({frame.segment_path for frame in unloaded_records}):
        capture_images = [
            image
            for frame, image in zip(unloaded_records, unloaded_images, strict=True)
            if frame.segment_path == capture_path
        ]
        capture_reference = unloaded_median_rgb(capture_images)
        capture_calibration = calibrate_analysis_strip(
            capture_reference,
            green_excess_threshold_dn=config.green_excess_threshold_dn,
        )
        y, x = np.nonzero(capture_calibration.reference_mask)
        capture_centroids.append(np.asarray((x.mean(), y.mean()), dtype=np.float64))
    capture_centroid_span = max(
        (
            float(np.linalg.norm(first - second))
            for position, first in enumerate(capture_centroids)
            for second in capture_centroids[position + 1 :]
        ),
        default=0.0,
    )
    calibration = calibrate_analysis_strip(
        reference,
        green_excess_threshold_dn=config.green_excess_threshold_dn,
    )
    canonical_reference = warp_rgb(reference, calibration).astype(np.float32)
    contour_reference = build_contour_reference(
        reference,
        calibration.reference_mask,
        sample_count=config.deformation_contour_samples,
        search_radius_px=config.deformation_search_radius_px,
    )
    unloaded_rows = [
        _unloaded_stability_row(index, frame, image, reference)
        for frame, image in zip(unloaded_records, unloaded_images, strict=True)
    ]
    frame_rows: list[dict[str, Any]] = []
    signatures: list[np.ndarray] = []
    for frame in loaded_records:
        image = _load_rgb(frame.rgb_path)
        canonical = warp_rgb(image, calibration)
        delta = canonical.astype(np.float32) - canonical_reference
        measurements = frame.measurements
        force = actual_force_magnitude(
            float(measurements["Fx_N"]),
            float(measurements["Fy_N"]),
            float(measurements["Fz_N"]),
        )
        metrics = optical_metrics(delta, canonical)
        deformation = contour_deformation(image, contour_reference)
        row = _frame_identity(index, frame)
        row.update(_measurement_values(measurements))
        row["actual_force_n"] = force
        row.update(metrics)
        row.update(deformation)
        for channel in "RGB":
            for metric in ("mae", "rms"):
                source = f"optical_{metric}_{channel}_dn"
                row[f"{source}_per_n"] = (
                    metrics[source] / force if force > 0.0 else float("nan")
                )
        deformation_rms = float(deformation["deformation_rms_px"])
        for metric in ("mae", "rms"):
            source = f"optical_{metric}_G_dn"
            row[f"{source}_per_deformation_px"] = (
                metrics[source] / deformation_rms
                if bool(deformation["deformation_valid"]) and deformation_rms > 0.0
                else float("nan")
            )
        frame_rows.append(row)
        signatures.append(longitudinal_signature(delta[:, :, 1], config.signature_bins))
    return {
        "frame_rows": frame_rows,
        "signatures": signatures,
        "unloaded_rows": unloaded_rows,
        "calibration_mask_area_px": int(np.count_nonzero(calibration.reference_mask)),
        "unloaded_capture_geometry_centroid_span_px": capture_centroid_span,
    }


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _frame_identity(index: SessionIndex, frame: Any) -> dict[str, Any]:
    run = frame.run
    return {
        "specimen_id": index.session.specimen_id,
        "material": index.session.material,
        "morphology": index.session.morphology,
        "run_id": run.run_id,
        "indenter": run.indenter,
        "hole_index": run.hole_index,
        "repetition_index": run.repetition_index,
        "target_force_n": frame.target_force_n,
        "frame_index": int(frame.measurements["frame_index"]),
        "image_path": str(frame.rgb_path),
    }


def _measurement_values(values: Any) -> dict[str, Any]:
    numeric = (
        "capture_elapsed_s",
        "camera_host_time_s",
        "camera_device_timestamp_ms",
        "camera_frame_number",
        "bota_host_time_s",
        "bota_sensor_timestamp",
        "camera_bota_time_delta_ms",
        "Fx_N",
        "Fy_N",
        "Fz_N",
        "Mx_Nm",
        "My_Nm",
        "Mz_Nm",
        "temperature_C",
        "bota_status",
    )
    return {name: float(values[name]) for name in numeric}


def _unloaded_stability_row(
    index: SessionIndex,
    frame: Any,
    image: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    delta = image.astype(np.float32) - reference.astype(np.float32)
    row = {
        "specimen_id": index.session.specimen_id,
        "material": index.session.material,
        "morphology": index.session.morphology,
        "capture_id": frame.segment_path.name,
        "frame_index": int(frame.measurements["frame_index"]),
        "image_path": str(frame.rgb_path),
        "mae_rgb_dn": float(np.mean(np.abs(delta))),
        "rms_rgb_dn": float(np.sqrt(np.mean(delta**2))),
    }
    for channel_index, channel in enumerate("RGB"):
        difference = delta[:, :, channel_index]
        row[f"mae_{channel}_dn"] = float(np.mean(np.abs(difference)))
        row[f"rms_{channel}_dn"] = float(np.sqrt(np.mean(difference**2)))
    return row


def _cache_name(index: SessionIndex) -> str:
    suffix = hashlib.sha1(str(index.path).encode()).hexdigest()[:8]
    return f"{index.session_id}_{suffix}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


__all__ = ["AnalysisConfig", "analyze_sessions"]

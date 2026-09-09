"""Corrected, traceable analysis for LUMO Figure 6 panels (a)--(c).

This module deliberately does not import or build the force time-series and
robustness panels.  Figure 5 retains its separate decoder and feature path.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable

import cv2
import h5py
import numpy as np

from algorithm.canonical import CanonicalFingerConfig, build_canonical_map
from algorithm.contact_localization import (
    ContactLocalizationConfig,
    UnloadedOpticalReference,
    localize_response_profile,
)
from algorithm.fingertip_segmentation import segment_fingertip
from algorithm.led_localization import LED_POSITIONS_MM
from experiments.analysis.fig5c_decoder import PaperFigureConfig, read_csv
from experiments.analysis.optical import load_rgb, temporal_median_rgb


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTACT_DATASET_H5 = (
    REPOSITORY_ROOT / "output" / "upload" / "contact_dataset.h5"
)
PROTOCOL_ID = "fig6c_corrected_5n_led_registered_response_v2"
OUTPUT_SUBDIRECTORY = PROTOCOL_ID
PROFILE_BIN_COUNT = 128
TARGET_FORCE_N = 5.0
INDENTER = "sphere_10mm"
LED_PITCH_MM = 11.0

HISTORY_ROOTS = {
    "solaris": REPOSITORY_ROOT / "output" / "analysis" / "solaris_contact_history",
    "dragon_skin": REPOSITORY_ROOT
    / "output"
    / "analysis"
    / "dragon_skin_contact_history",
}

# Frozen unloaded-image registration settings.  They are geometry/photometry
# settings, not values selected against contact labels.
ANCHOR_LOCAL_SIGMA_PX = 1.2
ANCHOR_BACKGROUND_SIGMA_PX = 12.0
ANCHOR_MINIMUM_PITCH_FRACTION = 0.06
ANCHOR_MAXIMUM_PITCH_FRACTION = 0.22
ANCHOR_PITCH_SAMPLE_COUNT = 200
ANCHOR_MIDPOINT_PENALTY = 0.35
ANCHOR_WEAKEST_TOOTH_WEIGHT = 10.0
ANCHOR_MINIMUM_TOOTH_CONTRAST_DN = 7.5
STORED_MASK_OVERLAP_THRESHOLD = 0.80


@dataclass(frozen=True)
class AnchorRegistration:
    """One unloaded-image LED registration and its fixed source remap."""

    calibration_index: int
    session_index: int
    capture_id: str
    source_mode: str
    valid: bool
    status: str
    map_x: np.ndarray
    map_y: np.ndarray
    support_mask: np.ndarray
    reference_rgb: np.ndarray
    led_rows: np.ndarray
    led_source_xy_px: np.ndarray
    tooth_contrast_dn: np.ndarray
    source_mask_overlap: float
    source_paths: tuple[str, ...]


@dataclass(frozen=True)
class ContactObservation:
    """One 5-N target contact episode in the shared Figure 6(c) profile space."""

    observation_id: str
    material: str
    morphology: str
    specimen_id: str
    run_id: str
    hole_index: int
    repetition_index: int
    calibration_index: int
    reference_id: str
    frame_count: int
    actual_force_median_n: float
    actual_force_std_n: float
    actual_force_min_n: float
    actual_force_max_n: float
    geometry_valid: bool
    contact_valid: bool
    geometry_status: str
    contact_status: str
    response_profile: np.ndarray
    evidence_profile: np.ndarray
    geometry_position_mm: float | None
    geometry_localization_valid: bool
    geometry_localization_status: str


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8")
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
            for value in values
        ]
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _morphology_metric_path(
    config: PaperFigureConfig,
    material: str,
    morphology: str,
    indenter: str,
) -> Path:
    return config.source_root(material, morphology, indenter) / "results" / "morphology_metrics.csv"


def _matching_metric_row(
    path: Path,
    material: str,
    morphology: str,
    indenter: str,
) -> dict[str, str] | None:
    for row in read_csv(path):
        if (
            row["material"] == material
            and row["morphology"] == morphology
            and row["indenter"] == indenter
        ):
            return row
    return None


def validated_recontact_ratio(
    d_neighbor_dn_per_n: float,
    w_recontact_dn_per_n: float,
    stored_ratio: float,
) -> tuple[float | None, str]:
    """Validate the established aggregate ``D_neighbor / W_recontact`` ratio."""

    if not np.isfinite(d_neighbor_dn_per_n) or d_neighbor_dn_per_n < 0.0:
        return None, "invalid aggregate D_neighbor numerator"
    if not np.isfinite(w_recontact_dn_per_n) or w_recontact_dn_per_n <= 0.0:
        return None, "missing or nonpositive W_recontact denominator"
    ratio = d_neighbor_dn_per_n / w_recontact_dn_per_n
    if not np.isfinite(stored_ratio) or not np.isclose(
        ratio, stored_ratio, rtol=1e-10, atol=1e-12
    ):
        return None, "stored Q does not equal aggregate D divided by aggregate W"
    return ratio, ""


def panel_b_values(config: PaperFigureConfig) -> list[dict[str, Any]]:
    """Load the existing aggregate D/W definition with explicit provenance."""

    output: list[dict[str, Any]] = []
    for material in config.materials:
        for indenter in config.indenters:
            for morphology in config.morphologies:
                path = _morphology_metric_path(config, material, morphology, indenter)
                row = _matching_metric_row(path, material, morphology, indenter) if path.is_file() else None
                status = "measured"
                reason = ""
                d_value: float | str = ""
                w_value: float | str = ""
                q_value: float | str = ""
                if row is None:
                    status = "unavailable"
                    reason = "missing matching morphology_metrics row"
                else:
                    try:
                        d_value = float(row["D_neighbor_median_DN_per_N"])
                        w_value = float(row["W_median_DN_per_N"])
                        stored_q = float(row["D_neighbor_over_W"])
                    except (KeyError, ValueError):
                        status = "unavailable"
                        reason = "non-numeric D_neighbor/W_recontact/Q source field"
                    else:
                        ratio, reason = validated_recontact_ratio(
                            d_value, w_value, stored_q
                        )
                        if ratio is None:
                            status = "unavailable"
                        else:
                            q_value = ratio
                output.append(
                    {
                        "material": material,
                        "material_display": config.material_labels[material],
                        "morphology_id": morphology,
                        "morphology_display": config.morphology_labels[morphology],
                        "indenter": indenter,
                        "indenter_display": config.indenter_labels[indenter],
                        "D_neighbor_DN_per_N": d_value,
                        "W_recontact_DN_per_N": w_value,
                        "Q_recontact": q_value,
                        "numerator_source_field": "D_neighbor_median_DN_per_N",
                        "denominator_source_field": "W_median_DN_per_N",
                        "stored_ratio_source_field": "D_neighbor_over_W",
                        "aggregation": "ratio of aggregate medians: D_neighbor_median / W_median",
                        "optical_representation": "linear slope of 128-bin longitudinal Green profile versus measured force",
                        "source_path": _display_path(path),
                        "source_override": str(
                            (material, morphology, indenter) in config.condition_overrides
                        ).lower(),
                        "status": status,
                        "failure_reason": reason,
                    }
                )
    return output


def _history_specimen_id(material: str, morphology: str) -> str:
    session = (
        REPOSITORY_ROOT
        / "output"
        / "contact_history"
        / f"2026-09-06_{material}_{morphology}"
        / "session.json"
    )
    if not session.is_file():
        return "unavailable"
    payload = json.loads(session.read_text(encoding="utf-8"))
    return str(payload["specimen"]["specimen_id"])


def _history_indenter_support(material: str, morphology: str) -> tuple[str, ...]:
    session = (
        REPOSITORY_ROOT
        / "output"
        / "contact_history"
        / f"2026-09-06_{material}_{morphology}"
    )
    values: set[str] = set()
    for path in sorted((session / "runs").glob("run_*/run.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            values.add(str(payload["indenter"]))
    return tuple(sorted(values))


def aggregate_run_first_cycle_values(
    rows: Iterable[dict[str, Any]],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Reduce branch/force cells within each run before comparing runs."""

    values = list(rows)
    run_ids = tuple(sorted({str(row["run_id"]) for row in values}))
    run_values = np.asarray(
        [
            np.median(
                [
                    float(row["W_cycle_dn"])
                    for row in values
                    if str(row["run_id"]) == run_id
                ]
            )
            for run_id in run_ids
        ],
        dtype=np.float64,
    )
    return run_ids, run_values


def aggregate_panel_a(
    config: PaperFigureConfig,
    recontact_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join run-first W_cycle and 10-mm W_recontact without duplicating data."""

    recontact = {
        (row["material"], row["indenter"], row["morphology_id"]): row
        for row in recontact_rows
    }
    output: list[dict[str, Any]] = []
    for material in config.materials:
        detail_path = HISTORY_ROOTS[material] / "results" / "same_contact_repeatability.csv"
        detail = read_csv(detail_path)
        for morphology in config.morphologies:
            selected = [row for row in detail if row["morphology"] == morphology]
            run_ids, run_values = aggregate_run_first_cycle_values(selected)
            indenters = _history_indenter_support(material, morphology)
            condition_indenter = indenters[0] if len(indenters) == 1 else ""
            diameter_mm: float | str = ""
            if condition_indenter.startswith("sphere_") and condition_indenter.endswith("mm"):
                diameter_mm = float(
                    condition_indenter.removeprefix("sphere_").removesuffix("mm")
                )
            source = recontact.get((material, condition_indenter, morphology))
            status = "measured"
            reason = ""
            w_recontact: float | str = ""
            if not selected or not len(run_values):
                status = "unavailable"
                reason = "no maintained-contact W_cycle observations"
            elif len(indenters) != 1:
                status = "unavailable"
                reason = f"maintained-contact indenter support is not unique: {indenters}"
            elif source is None or source["status"] != "measured":
                status = "unavailable"
                reason = "matched W_recontact condition unavailable"
            else:
                w_recontact = float(source["W_recontact_DN_per_N"])
            output.append(
                {
                    "material": material,
                    "material_display": config.material_labels[material],
                    "morphology_id": morphology,
                    "morphology_display": config.morphology_labels[morphology],
                    "indenter": condition_indenter,
                    "indenter_diameter_mm": diameter_mm,
                    "indenter_display": (
                        config.indenter_labels.get(condition_indenter, condition_indenter)
                    ),
                    "specimen_id": _history_specimen_id(material, morphology),
                    "specimen_count": 1 if selected else 0,
                    "source_run_count": len(run_ids),
                    "source_run_ids": ";".join(run_ids),
                    "contact_hole_support": ";".join(
                        str(value) for value in sorted({int(row["hole_index"]) for row in selected})
                    ),
                    "force_support_n": (
                        f"{min(float(row['actual_force_n']) for row in selected):g}-"
                        f"{max(float(row['actual_force_n']) for row in selected):g}"
                        if selected
                        else ""
                    ),
                    "W_cycle_DN": float(np.median(run_values)) if len(run_values) else "",
                    "W_recontact_DN_per_N": w_recontact,
                    "W_cycle_aggregation": "median branch/force W_cycle within run, then median across runs",
                    "W_recontact_aggregation": "median run-to-same-location slope-template RMS across six locations",
                    "W_cycle_source_path": _display_path(detail_path),
                    "W_recontact_source_path": "" if source is None else source["source_path"],
                    "support_note": (
                        "W_cycle uses maintained-contact holes 1/3/5 over its common branch-force support; "
                        "W_recontact uses independent re-contacts at holes 1-6 and slopes fitted across 2/5/10/15 N"
                    ),
                    "status": status,
                    "failure_reason": reason,
                }
            )
    return output


def _source_path_for_frame(
    data: h5py.File,
    frame_index: int,
    session_names: np.ndarray,
    session_paths: tuple[Path, ...],
) -> Path:
    session_index = int(data["frames/session_index"][frame_index])
    relative = Path(data["frames/source_path"][frame_index].decode("utf-8"))
    relative = relative.relative_to(str(session_names[session_index]))
    return session_paths[session_index] / relative


def _canonical_registration_map(
    unloaded_rgb: np.ndarray,
    stored_source_mask: np.ndarray,
    stored_map_x: np.ndarray,
    stored_map_y: np.ndarray,
    stored_support: np.ndarray,
) -> tuple[str, float, np.ndarray, np.ndarray, np.ndarray]:
    """Prefer the production fingertip map when it agrees with stored geometry."""

    source_mode = "stored_optical_strip"
    overlap = 0.0
    map_x = np.asarray(stored_map_x, dtype=np.float32)
    map_y = np.asarray(stored_map_y, dtype=np.float32)
    support = np.asarray(stored_support, dtype=bool)
    try:
        fingertip = segment_fingertip(unloaded_rgb)
        overlap = float(
            np.count_nonzero(fingertip.mask & stored_source_mask)
            / np.count_nonzero(fingertip.mask)
        )
        if overlap >= STORED_MASK_OVERLAP_THRESHOLD:
            canonical = build_canonical_map(
                fingertip,
                CanonicalFingerConfig(
                    longitudinal_samples=256,
                    transverse_samples=128,
                    transverse_inset_fraction=0.04,
                ),
            )
            source_mode = "algorithm_fingertip_map"
            map_x = canonical.map_x
            map_y = canonical.map_y
            support = np.ones(canonical.shape, dtype=bool)
    except RuntimeError:
        pass
    return source_mode, overlap, map_x, map_y, support


def detect_led_anchor_lattice(
    reference_rgb: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    support_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, str, bool]:
    """Detect a five-tooth LED lattice using unloaded Green contrast only.

    The row direction is fixed by the side-view convention: smaller source
    image y is distal.  Contact labels are not accepted by this function.
    """

    image = np.asarray(reference_rgb)
    map_x = np.asarray(map_x, dtype=np.float32)
    map_y = np.asarray(map_y, dtype=np.float32)
    support = np.asarray(support_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("reference_rgb must be H x W x 3 uint8")
    if map_x.shape != map_y.shape or support.shape != map_x.shape:
        raise ValueError("map and support arrays must have equal shape")
    if min(map_x.shape) < 8 or not np.any(support):
        raise ValueError("registration map has insufficient support")

    flipped = bool(np.nanmean(map_y[-1]) < np.nanmean(map_y[0]))
    if flipped:
        map_x = map_x[::-1]
        map_y = map_y[::-1]
        support = support[::-1]
    canonical = cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    green = canonical[:, :, 1].astype(np.float32)
    response = np.maximum(
        cv2.GaussianBlur(green, (0, 0), ANCHOR_LOCAL_SIGMA_PX)
        - cv2.GaussianBlur(green, (0, 0), ANCHOR_BACKGROUND_SIGMA_PX),
        0.0,
    )
    row_count = response.shape[0]
    best: tuple[float, np.ndarray, int, np.ndarray] | None = None
    for pitch in np.linspace(
        ANCHOR_MINIMUM_PITCH_FRACTION * row_count,
        ANCHOR_MAXIMUM_PITCH_FRACTION * row_count,
        ANCHOR_PITCH_SAMPLE_COUNT,
    ):
        start_count = max(2, int(row_count - 4.0 * pitch))
        for start in np.linspace(0.0, row_count - 1.0 - 4.0 * pitch, start_count):
            rows = np.rint(start + np.arange(5) * pitch).astype(np.int32)
            mid_rows = np.rint(start + (np.arange(4) + 0.5) * pitch).astype(np.int32)
            teeth = response[rows]
            middle = response[mid_rows]
            background = np.vstack(
                (middle[0], 0.5 * (middle[:-1] + middle[1:]), middle[-1])
            )
            contrast = teeth - ANCHOR_MIDPOINT_PENALTY * background
            valid_columns = np.all(support[rows], axis=0)
            weakest = np.min(contrast, axis=0)
            score = np.sum(contrast, axis=0) + ANCHOR_WEAKEST_TOOTH_WEIGHT * weakest
            score[~valid_columns] = -np.inf
            column = int(np.argmax(score))
            candidate_score = float(score[column])
            if best is None or candidate_score > best[0]:
                best = (candidate_score, rows, column, contrast[:, column])
    if best is None or not np.isfinite(best[0]):
        empty = np.full(5, np.nan)
        return empty, np.full((5, 2), np.nan), empty, False, "no supported five-LED candidate", flipped

    _, rows, column, tooth_contrast = best
    source_xy = np.column_stack((map_x[rows, column], map_y[rows, column]))
    valid = bool(
        np.all(np.isfinite(source_xy))
        and np.all(tooth_contrast >= ANCHOR_MINIMUM_TOOTH_CONTRAST_DN)
    )
    status = (
        "accepted: five unloaded-image contrast teeth pass fixed QC"
        if valid
        else "rejected: at least one unloaded-image lattice tooth fails contrast QC"
    )
    return rows.astype(np.float64), source_xy, tooth_contrast, valid, status, flipped


def _draw_anchor_overlay(
    path: Path,
    rgb: np.ndarray,
    source_mask: np.ndarray,
    registration: AnchorRegistration,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(
        np.asarray(source_mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (180, 180, 180), 1)
    color = (60, 200, 60) if registration.valid else (40, 40, 230)
    for index, point in enumerate(registration.led_source_xy_px, start=1):
        if not np.all(np.isfinite(point)):
            continue
        xy = tuple(int(round(value)) for value in point)
        cv2.circle(overlay, xy, 7, color, 2, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"LED{index}",
            (xy[0] + 8, xy[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        overlay,
        f"{registration.source_mode} | {registration.status}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), overlay):
        raise RuntimeError(f"failed to write registration overlay: {path}")


def load_anchor_registrations(
    artifact_path: Path,
    overlay_directory: Path,
) -> tuple[list[AnchorRegistration], list[dict[str, Any]]]:
    """Register every unloaded capture used by 10-mm, 5-N observations."""

    registrations: list[AnchorRegistration] = []
    anchor_rows: list[dict[str, Any]] = []
    with h5py.File(artifact_path, "r") as data:
        session_names = _decode(data["sessions/name"][:])
        session_paths = tuple(Path(value) for value in _decode(data["sessions/source_path"][:]))
        frame_calibration = np.asarray(data["frames/calibration_index"][:], dtype=np.int64)
        frame_unloaded = np.asarray(data["frames/unloaded"][:], dtype=bool)
        frame_indenter = _decode(data["frames/indenter"][:])
        frame_target = np.asarray(data["frames/target_force_n"][:], dtype=np.float64)
        used_calibrations = set(
            int(value)
            for value in np.unique(
                frame_calibration[
                    ~frame_unloaded
                    & (frame_indenter == INDENTER)
                    & np.isclose(frame_target, TARGET_FORCE_N)
                ]
            )
        )
        for calibration_index in range(int(data.attrs["calibration_count"])):
            if calibration_index not in used_calibrations:
                continue
            unloaded_indices = np.flatnonzero(
                frame_unloaded & (frame_calibration == calibration_index)
            )
            paths = tuple(
                _source_path_for_frame(data, int(index), session_names, session_paths)
                for index in unloaded_indices
            )
            images = [load_rgb(path) for path in paths]
            reference_rgb = temporal_median_rgb(images)
            stored_source_mask = np.asarray(
                data["calibrations/source_mask"][calibration_index], dtype=bool
            )
            source_mode, overlap, map_x, map_y, support = _canonical_registration_map(
                reference_rgb,
                stored_source_mask,
                data["calibrations/map_x"][calibration_index],
                data["calibrations/map_y"][calibration_index],
                data["calibrations/support_mask_full"][calibration_index],
            )
            led_rows, source_xy, contrast, valid, status, flipped = detect_led_anchor_lattice(
                reference_rgb, map_x, map_y, support
            )
            if flipped:
                map_x = map_x[::-1].copy()
                map_y = map_y[::-1].copy()
                support = support[::-1].copy()
            session_index = int(data["calibrations/session_index"][calibration_index])
            capture_id = data["calibrations/capture_id"][calibration_index].decode("utf-8")
            registration = AnchorRegistration(
                calibration_index=calibration_index,
                session_index=session_index,
                capture_id=capture_id,
                source_mode=source_mode,
                valid=valid,
                status=status,
                map_x=np.asarray(map_x, dtype=np.float32),
                map_y=np.asarray(map_y, dtype=np.float32),
                support_mask=np.asarray(support, dtype=bool),
                reference_rgb=reference_rgb,
                led_rows=led_rows,
                led_source_xy_px=source_xy,
                tooth_contrast_dn=contrast,
                source_mask_overlap=overlap,
                source_paths=tuple(_display_path(path) for path in paths),
            )
            registrations.append(registration)
            _draw_anchor_overlay(
                overlay_directory / f"calibration_{calibration_index:03d}.png",
                reference_rgb,
                stored_source_mask,
                registration,
            )
            for led_index, (row, point, tooth) in enumerate(
                zip(led_rows, source_xy, contrast, strict=True), start=1
            ):
                anchor_rows.append(
                    {
                        "calibration_index": calibration_index,
                        "session_index": session_index,
                        "session_name": session_names[session_index],
                        "capture_id": capture_id,
                        "source_mode": source_mode,
                        "source_mask_overlap": overlap,
                        "registration_valid": valid,
                        "registration_status": status,
                        "led_index": led_index,
                        "led_position_mm": float(LED_POSITIONS_MM[led_index - 1]),
                        "canonical_row": float(row),
                        "canonical_row_fraction": float(row / (len(map_y) - 1)),
                        "source_x_px": float(point[0]),
                        "source_y_px": float(point[1]),
                        "tooth_contrast_dn": float(tooth),
                        "distal_orientation": "smaller source-image y",
                        "unloaded_source_paths": ";".join(registration.source_paths),
                        "overlay_path": _display_path(
                            overlay_directory / f"calibration_{calibration_index:03d}.png"
                        ),
                    }
                )
    return registrations, anchor_rows


def physical_coordinates_from_led_rows(
    row_coordinates: np.ndarray,
    led_rows: np.ndarray,
    led_positions_mm: np.ndarray = LED_POSITIONS_MM,
) -> np.ndarray:
    """Map canonical rows through measured anchors, including observed end support."""

    rows = np.asarray(row_coordinates, dtype=np.float64)
    anchors = np.asarray(led_rows, dtype=np.float64)
    positions = np.asarray(led_positions_mm, dtype=np.float64)
    if anchors.shape != positions.shape or anchors.ndim != 1:
        raise ValueError("LED rows and positions must be equal one-dimensional arrays")
    if not np.all(np.diff(anchors) > 0.0) or not np.all(np.diff(positions) > 0.0):
        raise ValueError("LED rows and positions must be distal-to-proximal")
    mapped = np.interp(rows, anchors, positions)
    below = rows < anchors[0]
    above = rows > anchors[-1]
    mapped[below] = positions[0] + (rows[below] - anchors[0]) * (
        (positions[1] - positions[0]) / (anchors[1] - anchors[0])
    )
    mapped[above] = positions[-1] + (rows[above] - anchors[-1]) * (
        (positions[-1] - positions[-2]) / (anchors[-1] - anchors[-2])
    )
    return mapped


def _warp_registration(rgb: np.ndarray, registration: AnchorRegistration) -> np.ndarray:
    return cv2.remap(
        np.asarray(rgb),
        registration.map_x,
        registration.map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _support_aware_response(
    current_rgb: np.ndarray,
    unloaded_rgb: np.ndarray,
    support_mask: np.ndarray,
    config: ContactLocalizationConfig,
) -> tuple[np.ndarray, float, float]:
    """Apply the production positive-Green brightest-fraction rule to support only."""

    current = np.asarray(current_rgb)
    unloaded = np.asarray(unloaded_rgb)
    support = np.asarray(support_mask, dtype=bool)
    if current.shape != unloaded.shape or current.shape[:2] != support.shape:
        raise ValueError("current/reference/support shapes must match")
    positive = np.maximum(
        current[:, :, 1].astype(np.float32) - unloaded[:, :, 1].astype(np.float32),
        0.0,
    )
    profile = np.full(len(positive), np.nan, dtype=np.float64)
    saturated_250: list[np.ndarray] = []
    saturated_255: list[np.ndarray] = []
    for row in range(len(positive)):
        columns = np.flatnonzero(support[row])
        if not len(columns):
            continue
        count = max(1, int(np.ceil(config.brightest_fraction * len(columns))))
        values = positive[row, columns]
        chosen = columns[np.argpartition(values, len(values) - count)[-count:]]
        profile[row] = float(np.mean(positive[row, chosen]))
        selected_green = current[row, chosen, 1]
        saturated_250.append(selected_green >= 250)
        saturated_255.append(selected_green >= 255)
    valid = np.isfinite(profile)
    if np.count_nonzero(valid) < 2:
        raise RuntimeError("registration has fewer than two supported response rows")
    indices = np.arange(len(profile))
    profile = np.interp(indices, indices[valid], profile[valid])
    if config.longitudinal_smoothing_sigma > 0.0:
        profile = cv2.GaussianBlur(
            profile.astype(np.float32)[:, None],
            (1, 0),
            sigmaX=0.0,
            sigmaY=config.longitudinal_smoothing_sigma,
            borderType=cv2.BORDER_REPLICATE,
        ).ravel()
    return (
        profile.astype(np.float64),
        float(np.mean(np.concatenate(saturated_250))),
        float(np.mean(np.concatenate(saturated_255))),
    )


def _condition_grid(
    registrations: Iterable[AnchorRegistration],
) -> np.ndarray | None:
    ranges = []
    for registration in registrations:
        if not registration.valid:
            continue
        supported_rows = np.flatnonzero(np.any(registration.support_mask, axis=1))
        physical = physical_coordinates_from_led_rows(
            supported_rows, registration.led_rows
        )
        ranges.append((float(np.min(physical)), float(np.max(physical))))
    if not ranges:
        return None
    lower = max(value[0] for value in ranges)
    upper = min(value[1] for value in ranges)
    if not np.isfinite(lower + upper) or upper <= lower:
        return None
    return np.linspace(lower, upper, PROFILE_BIN_COUNT)


def _regrid_profile(
    profile: np.ndarray,
    registration: AnchorRegistration,
    physical_grid_mm: np.ndarray,
) -> np.ndarray:
    supported_rows = np.flatnonzero(np.any(registration.support_mask, axis=1))
    source_mm = physical_coordinates_from_led_rows(supported_rows, registration.led_rows)
    values = np.asarray(profile, dtype=np.float64)[supported_rows]
    if physical_grid_mm[0] < source_mm[0] or physical_grid_mm[-1] > source_mm[-1]:
        raise RuntimeError("common grid extends beyond observed registration support")
    return np.interp(physical_grid_mm, source_mm, values)


def extract_contact_observations(
    artifact_path: Path,
    registrations: list[AnchorRegistration],
) -> tuple[list[ContactObservation], dict[tuple[str, str], np.ndarray]]:
    """Extract one unloaded-relative 5-N profile per independent contact run."""

    optical_config = ContactLocalizationConfig(unloaded_frame_count=2)
    registration_lookup = {
        registration.calibration_index: registration for registration in registrations
    }
    observations: list[ContactObservation] = []
    grids: dict[tuple[str, str], np.ndarray] = {}
    with h5py.File(artifact_path, "r") as data:
        session_names = _decode(data["sessions/name"][:])
        session_paths = tuple(Path(value) for value in _decode(data["sessions/source_path"][:]))
        session_material = _decode(data["sessions/material"][:])
        session_morphology = _decode(data["sessions/morphology"][:])
        session_specimen = _decode(data["sessions/specimen_id"][:])
        frame_session = np.asarray(data["frames/session_index"][:], dtype=np.int64)
        frame_calibration = np.asarray(data["frames/calibration_index"][:], dtype=np.int64)
        frame_unloaded = np.asarray(data["frames/unloaded"][:], dtype=bool)
        frame_indenter = _decode(data["frames/indenter"][:])
        frame_target = np.asarray(data["frames/target_force_n"][:], dtype=np.float64)
        frame_run = _decode(data["frames/run_id"][:])
        frame_hole = np.asarray(data["frames/hole_index"][:], dtype=np.int64)
        frame_repetition = np.asarray(data["frames/repetition_index"][:], dtype=np.int64)
        frame_force = np.asarray(data["frames/actual_force_n"][:], dtype=np.float64)

        for session_index in range(len(session_names)):
            key = (session_material[session_index], session_morphology[session_index])
            used = sorted(
                {
                    int(value)
                    for value in frame_calibration[
                        (frame_session == session_index)
                        & ~frame_unloaded
                        & (frame_indenter == INDENTER)
                        & np.isclose(frame_target, TARGET_FORCE_N)
                    ]
                }
            )
            grid = _condition_grid(
                registration_lookup[value]
                for value in used
                if value in registration_lookup
            )
            if grid is not None:
                grids[key] = grid

        reference_cache: dict[
            tuple[int, str, str], tuple[UnloadedOpticalReference, np.ndarray]
        ] = {}

        def reference_for(
            calibration_index: int,
            grid: np.ndarray,
        ) -> tuple[UnloadedOpticalReference, np.ndarray]:
            cache_key = (calibration_index, f"{grid[0]:.12g}", f"{grid[-1]:.12g}")
            if cache_key in reference_cache:
                return reference_cache[cache_key]
            registration = registration_lookup[calibration_index]
            reference_canonical = _warp_registration(
                registration.reference_rgb, registration
            )
            unloaded_indices = np.flatnonzero(
                frame_unloaded & (frame_calibration == calibration_index)
            )
            if len(unloaded_indices) < optical_config.unloaded_frame_count:
                raise RuntimeError(
                    f"calibration {calibration_index} has fewer than two unloaded frames"
                )
            profiles = []
            for index in unloaded_indices:
                rgb = load_rgb(
                    _source_path_for_frame(
                        data, int(index), session_names, session_paths
                    )
                )
                canonical = _warp_registration(rgb, registration)
                response, _, _ = _support_aware_response(
                    canonical,
                    reference_canonical,
                    registration.support_mask,
                    optical_config,
                )
                profiles.append(_regrid_profile(response, registration, grid))
            values = np.asarray(profiles, dtype=np.float64)
            center = np.median(values, axis=0)
            mad = np.median(np.abs(values - center[None, :]), axis=0)
            sigma = np.maximum(1.4826 * mad, optical_config.noise_floor_dn)
            threshold = center + optical_config.noise_sigma_multiplier * sigma
            reference = UnloadedOpticalReference(
                canonical_rgb=np.zeros((len(grid), 1, 3), dtype=np.float32),
                response_center_dn=center,
                response_sigma_dn=sigma,
                response_threshold_dn=threshold,
                frame_count=len(values),
            )
            reference_cache[cache_key] = (reference, reference_canonical)
            return reference, reference_canonical

        selected = np.flatnonzero(
            ~frame_unloaded
            & (frame_indenter == INDENTER)
            & np.isclose(frame_target, TARGET_FORCE_N)
        )
        groups: dict[tuple[int, str, int, int], list[int]] = {}
        for index in selected:
            key = (
                int(frame_session[index]),
                str(frame_run[index]),
                int(frame_hole[index]),
                int(frame_repetition[index]),
            )
            groups.setdefault(key, []).append(int(index))

        for (session_index, run_id, hole, repetition), indices in sorted(groups.items()):
            material = str(session_material[session_index])
            morphology = str(session_morphology[session_index])
            condition = (material, morphology)
            calibration_indices = np.unique(frame_calibration[indices])
            if len(calibration_indices) != 1:
                raise RuntimeError(f"{run_id} maps to multiple unloaded captures")
            calibration_index = int(calibration_indices[0])
            registration = registration_lookup[calibration_index]
            grid = grids.get(condition)
            observation_id = f"{session_names[session_index]}:{run_id}:5N"
            geometry_valid = registration.valid and grid is not None
            geometry_status = registration.status if grid is not None else "no common observed physical support"
            response = np.full(PROFILE_BIN_COUNT, np.nan, dtype=np.float64)
            evidence = np.full(PROFILE_BIN_COUNT, np.nan, dtype=np.float64)
            contact_valid = False
            contact_status = "unavailable: geometry registration failed"
            geometry_position: float | None = None
            geometry_localization_valid = False
            geometry_localization_status = "unavailable: geometry registration failed"

            if geometry_valid and grid is not None:
                reference, reference_canonical = reference_for(calibration_index, grid)
                loaded_images = [
                    load_rgb(
                        _source_path_for_frame(
                            data, index, session_names, session_paths
                        )
                    )
                    for index in indices
                ]
                current_rgb = temporal_median_rgb(loaded_images)
                current_canonical = _warp_registration(current_rgb, registration)
                native_response, saturation_250, saturation_255 = _support_aware_response(
                    current_canonical,
                    reference_canonical,
                    registration.support_mask,
                    optical_config,
                )
                response = _regrid_profile(native_response, registration, grid)
                led_coordinates = (LED_POSITIONS_MM - grid[0]) / (grid[-1] - grid[0])
                result = localize_response_profile(
                    response,
                    reference,
                    led_coordinates,
                    optical_config,
                    saturation_fraction_ge_250=saturation_250,
                    saturation_fraction_eq_255=saturation_255,
                )
                evidence = result.evidence_profile
                saturation_valid = (
                    saturation_250 <= optical_config.maximum_saturation_fraction_ge_250
                    and saturation_255 <= optical_config.maximum_saturation_fraction_eq_255
                )
                contact_valid = result.contact_detected and saturation_valid
                contact_status = (
                    "valid contact evidence"
                    if contact_valid
                    else (
                        "invalid: saturation QC"
                        if not saturation_valid
                        else "invalid: response does not exceed unloaded-noise gate"
                    )
                )
                geometry_localization_valid = bool(
                    result.valid and result.position_mm is not None
                )
                geometry_localization_status = result.status
                geometry_position = result.position_mm

            force = frame_force[indices]
            observations.append(
                ContactObservation(
                    observation_id=observation_id,
                    material=material,
                    morphology=morphology,
                    specimen_id=str(session_specimen[session_index]),
                    run_id=run_id,
                    hole_index=hole,
                    repetition_index=repetition,
                    calibration_index=calibration_index,
                    reference_id=f"calibration_{calibration_index:03d}",
                    frame_count=len(indices),
                    actual_force_median_n=float(np.median(force)),
                    actual_force_std_n=float(np.std(force)),
                    actual_force_min_n=float(np.min(force)),
                    actual_force_max_n=float(np.max(force)),
                    geometry_valid=geometry_valid,
                    contact_valid=contact_valid,
                    geometry_status=geometry_status,
                    contact_status=contact_status,
                    response_profile=response,
                    evidence_profile=evidence,
                    geometry_position_mm=geometry_position,
                    geometry_localization_valid=geometry_localization_valid,
                    geometry_localization_status=geometry_localization_status,
                )
            )
    return observations, grids


def write_observations_npz(
    path: Path,
    observations: list[ContactObservation],
    grids: dict[tuple[str, str], np.ndarray],
) -> Path:
    """Serialize the common response/evidence arrays and companion metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = {
        "observation_id": np.asarray([item.observation_id for item in observations]),
        "material": np.asarray([item.material for item in observations]),
        "morphology": np.asarray([item.morphology for item in observations]),
        "specimen_id": np.asarray([item.specimen_id for item in observations]),
        "run_id": np.asarray([item.run_id for item in observations]),
        "hole_index": np.asarray([item.hole_index for item in observations], dtype=np.int16),
        "repetition_index": np.asarray([item.repetition_index for item in observations], dtype=np.int16),
        "calibration_index": np.asarray([item.calibration_index for item in observations], dtype=np.int16),
        "reference_id": np.asarray([item.reference_id for item in observations]),
        "frame_count": np.asarray([item.frame_count for item in observations], dtype=np.int16),
        "actual_force_median_n": np.asarray([item.actual_force_median_n for item in observations]),
        "actual_force_std_n": np.asarray([item.actual_force_std_n for item in observations]),
        "actual_force_min_n": np.asarray([item.actual_force_min_n for item in observations]),
        "actual_force_max_n": np.asarray([item.actual_force_max_n for item in observations]),
        "geometry_valid": np.asarray([item.geometry_valid for item in observations]),
        "contact_valid": np.asarray([item.contact_valid for item in observations]),
        "geometry_status": np.asarray([item.geometry_status for item in observations]),
        "contact_status": np.asarray([item.contact_status for item in observations]),
        "geometry_localization_valid": np.asarray(
            [item.geometry_localization_valid for item in observations]
        ),
        "geometry_localization_status": np.asarray(
            [item.geometry_localization_status for item in observations]
        ),
        "response_profiles": np.asarray([item.response_profile for item in observations]),
        "evidence_profiles": np.asarray([item.evidence_profile for item in observations]),
        "physical_coordinate_mm": np.asarray(
            [
                grids.get(
                    (item.material, item.morphology),
                    np.full(PROFILE_BIN_COUNT, np.nan),
                )
                for item in observations
            ]
        ),
        "feature_protocol": np.asarray(PROTOCOL_ID),
        "target_force_n": np.asarray(TARGET_FORCE_N),
    }
    np.savez_compressed(path, **fields)
    return path


def enumerate_repetition_splits(
    repetitions: Iterable[int],
    maximum_calibration_contacts: int = 4,
) -> list[dict[str, Any]]:
    """Enumerate deterministic outer holdouts and inner calibration subsets."""

    values = tuple(sorted(set(int(value) for value in repetitions)))
    output: list[dict[str, Any]] = []
    for held_out in values:
        pool = tuple(value for value in values if value != held_out)
        output.append(
            {
                "outer_repetition": held_out,
                "calibration_contacts_per_location": 0,
                "calibration_repetitions": (),
                "test_repetitions": (held_out,),
            }
        )
        for count in range(1, min(maximum_calibration_contacts, len(pool)) + 1):
            for selected in combinations(pool, count):
                output.append(
                    {
                        "outer_repetition": held_out,
                        "calibration_contacts_per_location": count,
                        "calibration_repetitions": tuple(selected),
                        "test_repetitions": (held_out,),
                    }
                )
    return output


def _predict_template(
    feature: np.ndarray,
    templates: dict[int, np.ndarray],
) -> tuple[int, float]:
    distances = [
        (float(np.sqrt(np.mean((feature - template) ** 2))), hole)
        for hole, template in templates.items()
    ]
    distance, hole = min(distances)
    return hole, distance


def _is_true(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def summarize_optional_calibration_predictions(
    predictions: list[dict[str, Any]],
    config: PaperFigureConfig,
) -> list[dict[str, Any]]:
    """Reproduce the panel-(c) summary from per-contact predictions."""

    summary: list[dict[str, Any]] = []
    for material in config.materials:
        for morphology in config.morphologies:
            for count in range(config.maximum_calibration_contacts + 1):
                condition_rows = [
                    row
                    for row in predictions
                    if row["material"] == material
                    and row["morphology_id"] == morphology
                    and int(row["calibration_contacts_per_location"]) == count
                ]
                observations = {
                    str(row["observation_id"]): row for row in condition_rows
                }
                contact_failures: dict[str, int] = {}
                for row in observations.values():
                    if _is_true(row["contact_valid"]):
                        continue
                    reason = str(row["contact_status"])
                    contact_failures[reason] = contact_failures.get(reason, 0) + 1

                outer_values: list[dict[str, float]] = []
                for outer in sorted(
                    {int(row["outer_repetition"]) for row in condition_rows}
                ):
                    outer_rows = [
                        row
                        for row in condition_rows
                        if int(row["outer_repetition"]) == outer
                    ]
                    split_values = []
                    for split_id in sorted({row["split_id"] for row in outer_rows}):
                        split_rows = [
                            row for row in outer_rows if row["split_id"] == split_id
                        ]
                        valid_errors = [
                            float(row["absolute_error_mm"])
                            for row in split_rows
                            if _is_true(row["valid_prediction"])
                        ]
                        split_values.append(
                            {
                                "coverage": (
                                    len(valid_errors) / len(split_rows)
                                    if split_rows
                                    else 0.0
                                ),
                                "mae": (
                                    float(np.mean(valid_errors))
                                    if valid_errors
                                    else float("nan")
                                ),
                            }
                        )
                    coverage = (
                        float(np.mean([value["coverage"] for value in split_values]))
                        if split_values
                        else 0.0
                    )
                    maes = [
                        value["mae"]
                        for value in split_values
                        if np.isfinite(value["mae"])
                    ]
                    outer_values.append(
                        {
                            "coverage": coverage,
                            "mae": float(np.mean(maes)) if maes else float("nan"),
                        }
                    )

                outer_mae = np.asarray(
                    [
                        value["mae"]
                        for value in outer_values
                        if np.isfinite(value["mae"])
                    ],
                    dtype=np.float64,
                )
                outer_coverage = np.asarray(
                    [value["coverage"] for value in outer_values], dtype=np.float64
                )
                attempted_predictions = len(condition_rows)
                valid_predictions = sum(
                    _is_true(row["valid_prediction"]) for row in condition_rows
                )
                localization_failures: dict[str, int] = {}
                for row in condition_rows:
                    if _is_true(row["valid_prediction"]):
                        continue
                    reason = str(row["failure_reason"])
                    localization_failures[reason] = (
                        localization_failures.get(reason, 0) + 1
                    )
                measured = bool(len(outer_mae))
                contact_count = sum(
                    _is_true(row["contact_valid"]) for row in observations.values()
                )
                geometry_count: int | str = ""
                template_count: int | str = ""
                if count == 0:
                    geometry_count = sum(
                        _is_true(row["geometry_localization_valid"])
                        for row in observations.values()
                    )
                else:
                    template_count = sum(
                        _is_true(row["template_localization_valid"])
                        for row in condition_rows
                    )
                mean_mae: float | str = (
                    float(np.mean(outer_mae)) if measured else ""
                )
                median_mae: float | str = (
                    float(np.median(outer_mae)) if measured else ""
                )
                q25_mae: float | str = (
                    float(np.percentile(outer_mae, 25.0)) if measured else ""
                )
                q75_mae: float | str = (
                    float(np.percentile(outer_mae, 75.0)) if measured else ""
                )
                localization_coverage = (
                    float(np.mean(outer_coverage)) if len(outer_coverage) else 0.0
                )
                summary.append(
                    {
                        "material": material,
                        "material_display": config.material_labels[material],
                        "morphology_id": morphology,
                        "morphology_display": config.morphology_labels[morphology],
                        "indenter": INDENTER,
                        "calibration_contacts_per_location": count,
                        "inference_regime": (
                            "geometry centroid, no labeled contact examples"
                            if count == 0
                            else "nearest mean registered-response template"
                        ),
                        "outer_repetition_count": len(outer_values),
                        "attempted_test_contacts": len(observations),
                        "valid_contact_count": contact_count,
                        "contact_coverage": (
                            contact_count / len(observations) if observations else 0.0
                        ),
                        "geometry_valid_count": geometry_count,
                        "template_valid_count": template_count,
                        "prediction_attempt_count": attempted_predictions,
                        "valid_prediction_count": valid_predictions,
                        "localization_coverage": localization_coverage,
                        "localization_coverage_min": (
                            float(np.min(outer_coverage))
                            if len(outer_coverage)
                            else 0.0
                        ),
                        "mae_mm_given_valid": mean_mae,
                        "median_error_mm_given_valid": median_mae,
                        "q25_error_mm": q25_mae,
                        "q75_error_mm": q75_mae,
                        "contact_failure_counts_by_reason": json.dumps(
                            contact_failures, sort_keys=True
                        ),
                        "localization_failure_counts_by_reason": json.dumps(
                            localization_failures, sort_keys=True
                        ),
                        # Figure-facing aliases retain the established renderer contract.
                        "coverage_mean": localization_coverage,
                        "coverage_min": (
                            float(np.min(outer_coverage))
                            if len(outer_coverage)
                            else 0.0
                        ),
                        "localization_mae_mean_mm": mean_mae,
                        "localization_mae_median_mm": median_mae,
                        "localization_mae_q25_mm": q25_mae,
                        "localization_mae_q75_mm": q75_mae,
                        "uncertainty_definition": "quartiles across outer held-out repetitions after averaging calibration subsets",
                        "feature_protocol": PROTOCOL_ID,
                        "status": "measured" if measured else "unavailable",
                        "failure_reason": (
                            "" if measured else "no valid predictions in any outer repetition"
                        ),
                    }
                )
    return summary


def evaluate_optional_calibration(
    observations: list[ContactObservation],
    config: PaperFigureConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate the unchanged geometry decoder and response templates."""

    predictions: list[dict[str, Any]] = []
    split_records: list[dict[str, Any]] = []
    for material in config.materials:
        for morphology in config.morphologies:
            selected = [
                item
                for item in observations
                if item.material == material and item.morphology == morphology
            ]
            repetitions = sorted({item.repetition_index for item in selected})
            splits = enumerate_repetition_splits(
                repetitions, config.maximum_calibration_contacts
            )
            by_hole_repetition = {
                (item.hole_index, item.repetition_index): item for item in selected
            }
            for split_index, split in enumerate(splits):
                outer = int(split["outer_repetition"])
                count = int(split["calibration_contacts_per_location"])
                calibration_repetitions = tuple(split["calibration_repetitions"])
                split_id = (
                    f"{material}:{morphology}:outer{outer}:k{count}:"
                    + ("none" if not calibration_repetitions else "-".join(map(str, calibration_repetitions)))
                )
                test = sorted(
                    (item for item in selected if item.repetition_index == outer),
                    key=lambda item: item.hole_index,
                )
                templates: dict[int, np.ndarray] = {}
                template_failure = ""
                training_ids: list[str] = []
                if count > 0:
                    for hole in config.contact_positions_mm:
                        training: list[ContactObservation] = []
                        for repetition in calibration_repetitions:
                            observation = by_hole_repetition.get((hole, repetition))
                            if observation is None:
                                template_failure = (
                                    f"missing calibration episode for hole {hole}, repetition {repetition}"
                                )
                                break
                            training_ids.append(observation.observation_id)
                            if not observation.contact_valid:
                                template_failure = (
                                    f"invalid calibration episode {observation.observation_id}; budget not replaced"
                                )
                                break
                            if not np.all(np.isfinite(observation.response_profile)):
                                template_failure = (
                                    f"non-finite registered response for {observation.observation_id}"
                                )
                                break
                            training.append(observation)
                        if template_failure:
                            break
                        templates[hole] = np.mean(
                            np.asarray([item.response_profile for item in training]),
                            axis=0,
                        )

                for item in test:
                    true_mm = float(config.contact_positions_mm[item.hole_index])
                    valid = False
                    predicted_mm: float | str = ""
                    predicted_hole: int | str = ""
                    error_mm: float | str = ""
                    decoder_distance: float | str = ""
                    failure_reason = ""
                    template_localization_valid: bool | str = ""
                    if count == 0:
                        if (
                            item.geometry_localization_valid
                            and item.geometry_position_mm is not None
                        ):
                            valid = True
                            predicted_mm = float(item.geometry_position_mm)
                            error_mm = abs(float(predicted_mm) - true_mm)
                        else:
                            failure_reason = item.geometry_localization_status
                    elif template_failure:
                        template_localization_valid = False
                        failure_reason = template_failure
                    elif not item.contact_valid:
                        template_localization_valid = False
                        failure_reason = item.contact_status
                    elif not np.all(np.isfinite(item.response_profile)):
                        template_localization_valid = False
                        failure_reason = "non-finite registered response"
                    else:
                        predicted_hole, decoder_distance = _predict_template(
                            item.response_profile, templates
                        )
                        predicted_mm = float(config.contact_positions_mm[int(predicted_hole)])
                        error_mm = abs(float(predicted_mm) - true_mm)
                        template_localization_valid = True
                        valid = True
                    predictions.append(
                        {
                            "material": material,
                            "morphology_id": morphology,
                            "indenter": INDENTER,
                            "target_force_n": TARGET_FORCE_N,
                            "inference_regime": (
                                "geometry centroid, no labeled contact examples"
                                if count == 0
                                else "nearest mean full-profile template"
                            ),
                            "calibration_contacts_per_location": count,
                            "outer_repetition": outer,
                            "calibration_repetitions": ";".join(map(str, calibration_repetitions)),
                            "split_id": split_id,
                            "observation_id": item.observation_id,
                            "test_run_id": item.run_id,
                            "test_repetition": item.repetition_index,
                            "true_hole_index": item.hole_index,
                            "true_position_mm": true_mm,
                            "predicted_hole_index": predicted_hole,
                            "predicted_position_mm": predicted_mm,
                            "absolute_error_mm": error_mm,
                            "decoder_distance_rms_dn": decoder_distance,
                            "valid_prediction": valid,
                            "geometry_valid": item.geometry_valid,
                            "contact_valid": item.contact_valid,
                            "contact_status": item.contact_status,
                            "geometry_localization_valid": (
                                item.geometry_localization_valid
                            ),
                            "template_localization_valid": template_localization_valid,
                            "failure_reason": failure_reason,
                            "feature_observation_id": item.observation_id,
                            "contact_detection_feature": (
                                "thresholded unloaded-noise evidence_profile"
                            ),
                            "calibrated_decoding_feature": (
                                "pre-threshold registered response_profile"
                                if count > 0
                                else "not applicable"
                            ),
                            "feature_protocol": PROTOCOL_ID,
                        }
                    )
                split_records.append(
                    {
                        "split_id": split_id,
                        "material": material,
                        "morphology_id": morphology,
                        "outer_repetition": outer,
                        "calibration_contacts_per_location": count,
                        "calibration_repetitions": list(calibration_repetitions),
                        "training_observation_ids": sorted(set(training_ids)),
                        "test_observation_ids": [item.observation_id for item in test],
                        "template_failure": template_failure,
                    }
                )

    summary = summarize_optional_calibration_predictions(predictions, config)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "outer_split": "hold out one repetition across all six locations",
        "calibration_pool": "remaining four repetitions",
        "subset_rule": "enumerate every k-repetition subset without replacement",
        "calibration_contact_definition": "one independent contact episode per location",
        "contact_detection_feature": "thresholded unloaded-noise evidence_profile",
        "calibrated_decoding_feature": "pre-threshold registered response_profile",
        "split_records": split_records,
    }
    return predictions, summary, manifest


def _relevant_raw_input_digest(artifact_path: Path) -> tuple[str, int]:
    """Hash raw unloaded and 10-mm/5-N source images used by panel (c)."""

    digest = hashlib.sha256()
    count = 0
    with h5py.File(artifact_path, "r") as data:
        session_names = _decode(data["sessions/name"][:])
        session_paths = tuple(Path(value) for value in _decode(data["sessions/source_path"][:]))
        unloaded = np.asarray(data["frames/unloaded"][:], dtype=bool)
        indenter = _decode(data["frames/indenter"][:])
        target = np.asarray(data["frames/target_force_n"][:], dtype=np.float64)
        selected = np.flatnonzero(
            unloaded | ((indenter == INDENTER) & np.isclose(target, TARGET_FORCE_N))
        )
        for index in selected:
            path = _source_path_for_frame(data, int(index), session_names, session_paths)
            digest.update(_display_path(path).encode("utf-8"))
            digest.update(bytes.fromhex(_sha256(path)))
            count += 1
    return digest.hexdigest(), count


def _protocol_fingerprint(
    config_path: Path,
    artifact_path: Path,
    input_paths: Iterable[Path],
    raw_digest: str,
) -> tuple[str, dict[str, str]]:
    paths = {
        "config": config_path.resolve(),
        "contact_dataset_h5": artifact_path.resolve(),
        "fig6abc_source": Path(__file__).resolve(),
        "contact_localization_source": (
            REPOSITORY_ROOT / "algorithm" / "contact_localization.py"
        ).resolve(),
        "canonical_source": (REPOSITORY_ROOT / "algorithm" / "canonical.py").resolve(),
        "led_localization_source": (
            REPOSITORY_ROOT / "algorithm" / "led_localization.py"
        ).resolve(),
        "optical_source": (
            REPOSITORY_ROOT / "experiments" / "analysis" / "optical.py"
        ).resolve(),
    }
    for index, path in enumerate(sorted(set(Path(value).resolve() for value in input_paths))):
        paths[f"analysis_input_{index:02d}"] = path
    hashes = {name: _sha256(path) for name, path in paths.items()}
    hashes["raw_panel_c_image_set"] = raw_digest
    payload = json.dumps(
        {"protocol_id": PROTOCOL_ID, "hashes": hashes}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), hashes


def _write_analysis_notes(path: Path) -> Path:
    text = f"""# Corrected Figure 6(c) analysis notes

Protocol identifier: `{PROTOCOL_ID}`.

## Panel (a): contact-state variability

`W_cycle [DN]` is the median across runs after first taking the median of the
branch- and force-resolved same-contact cycle variability inside each run.
`W_recontact [DN/N]` is the existing median RMS deviation of independently
re-established-contact slope profiles from the same-location median template.
The maintained-contact sessions contain only the 10 mm sphere and holes 1, 3,
and 5.  The re-contact experiment contains holes 1--6 and uses slopes fitted
over the 2, 5, 10, and 15 N states.  The two axes are complementary metrics,
not two measurements of one quantity.

## Panel (b): re-contact distinguishability

`Q_recontact = D_neighbor / W_recontact`.  Both inputs are aggregate medians in
DN/N from the same 128-bin longitudinal Green slope-profile representation.
The ratio is descriptive and dimensionless; it is not held-out classification
accuracy.  Missing or nonpositive denominators remain unavailable.

## Panel (c): optional calibration

Every observation is one contact episode at the 5 N acquisition target,
formed by a temporal median of all frames in that hold.  The associated
unloaded capture defines a fixed remap and an unloaded noise reference.
Positive Green change is reduced row-wise using the brightest 10% of supported
pixels and Gaussian smoothing with sigma 2 rows.  This pre-threshold,
LED-registered 128-bin response profile is the common optical observation.

Contact validity retains the thresholded evidence profile, total evidence,
peak SNR, and saturation checks from the unloaded noise model.  At k=0, the
unchanged geometry-prior decoder uses the thresholded-evidence centroid and
maps it through five unloaded-image LED anchors at 0, 11, 22, 33, and 44 mm.
No contact labels, grid snapping, or fitted scale/offset enter this inference.
At k=1--4, one mean pre-threshold response-profile template is made per labeled
fixture location and the nearest template is reported at its known grid
coordinate.  Geometry-localization validity is not required for this calibrated
decoder.  These are two explicit decoding regimes under one observation
protocol.

The outer split holds out one complete repetition across all six locations.
Every k-subset of the remaining four repetitions is evaluated.  Subset metrics
are averaged inside an outer repetition; displayed quartiles are across outer
repetitions and are not confidence intervals.  MAE is conditional on a valid
prediction, and coverage is reported separately.

"No contact calibration" means no labeled contact examples.  It still uses an
unloaded optical reference and unloaded-image geometry registration.
"""
    path.write_text(text, encoding="utf-8")
    return path


def run_analysis(
    config: PaperFigureConfig,
    config_path: Path,
    *,
    artifact_path: Path = DEFAULT_CONTACT_DATASET_H5,
) -> dict[str, Path]:
    """Recompute and version every corrected panel (a)--(c) artifact."""

    artifact = artifact_path.resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    output = config.analysis_output_directory / OUTPUT_SUBDIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    overlays = output / "fig6c_geometry_overlays"

    b_rows = panel_b_values(config)
    a_rows = aggregate_panel_a(config, b_rows)
    registrations, anchor_rows = load_anchor_registrations(artifact, overlays)
    observations, grids = extract_contact_observations(artifact, registrations)
    predictions, c_summary, split_manifest = evaluate_optional_calibration(
        observations, config
    )
    panel_a_plot_rows = [
        {
            "material": row["material"],
            "material_display": row["material_display"],
            "morphology_id": row["morphology_id"],
            "morphology_display": row["morphology_display"],
            "indenter": row["indenter"],
            "indenter_diameter_mm": row["indenter_diameter_mm"],
            "W_cycle_DN": row["W_cycle_DN"],
            "W_recontact_DN_per_N": row["W_recontact_DN_per_N"],
            "status": row["status"],
        }
        for row in a_rows
        if row["status"] == "measured"
    ]

    paths = {
        "panel_a_details": _write_csv(
            output / "fig6a_variability_values.csv", a_rows
        ),
        "panel_a": _write_csv(
            output / "fig6a_contact_state_variability.csv", panel_a_plot_rows
        ),
        "panel_b": _write_csv(output / "fig6b_distinguishability_values.csv", b_rows),
        "observations": write_observations_npz(
            output / "fig6c_contact_observations.npz", observations, grids
        ),
        "anchors": _write_csv(output / "fig6c_geometry_anchors.csv", anchor_rows),
        "predictions": _write_csv(output / "fig6c_predictions.csv", predictions),
        "summary": _write_csv(output / "fig6c_summary.csv", c_summary),
        "notes": _write_analysis_notes(output / "fig6c_analysis_notes.md"),
    }
    split_path = output / "fig6c_split_manifest.json"
    split_path.write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
    paths["splits"] = split_path

    analysis_inputs = [
        HISTORY_ROOTS[material] / "results" / "same_contact_repeatability.csv"
        for material in config.materials
    ]
    analysis_inputs.extend(
        _morphology_metric_path(config, material, morphology, indenter)
        for material in config.materials
        for indenter in config.indenters
        for morphology in config.morphologies
    )
    raw_digest, raw_count = _relevant_raw_input_digest(artifact)
    fingerprint, hashes = _protocol_fingerprint(
        config_path, artifact, analysis_inputs, raw_digest
    )
    valid_registrations = sum(registration.valid for registration in registrations)
    provenance = {
        "protocol_id": PROTOCOL_ID,
        "protocol_fingerprint": fingerprint,
        "git_revision": _git_revision(),
        "input_hashes": hashes,
        "raw_panel_c_image_count": raw_count,
        "feature_definition": {
            "target_force_n": TARGET_FORCE_N,
            "indenter": INDENTER,
            "within_hold_aggregation": "pixelwise temporal median across all hold frames",
            "channel": "positive Green change from associated unloaded reference",
            "brightest_fraction": ContactLocalizationConfig().brightest_fraction,
            "longitudinal_smoothing_sigma_rows": ContactLocalizationConfig().longitudinal_smoothing_sigma,
            "noise_sigma_multiplier": ContactLocalizationConfig().noise_sigma_multiplier,
            "profile_bins": PROFILE_BIN_COUNT,
            "contact_detection_feature": (
                "thresholded evidence from the per-row unloaded noise model"
            ),
            "calibrated_decoding_feature": (
                "pre-threshold registered response_profile"
            ),
            "k0_localization_feature": (
                "unchanged thresholded-evidence geometry centroid"
            ),
        },
        "geometry_reference_protocol": {
            "orientation": "LED1 distal; smaller source-image y is distal",
            "physical_led_positions_mm": LED_POSITIONS_MM.tolist(),
            "lattice_pitch_mm": LED_PITCH_MM,
            "contact_labels_used_for_registration": False,
            "accepted_registration_count": int(valid_registrations),
            "attempted_registration_count": len(registrations),
            "overlay_directory": _display_path(overlays),
        },
        "sample_counts": {
            "contact_episodes": len(observations),
            "specimens": len({item.specimen_id for item in observations}),
            "runs": len({(item.specimen_id, item.run_id) for item in observations}),
            "by_condition": [
                {
                    "material": material,
                    "morphology": morphology,
                    "contact_episodes": sum(
                        item.material == material and item.morphology == morphology
                        for item in observations
                    ),
                    "runs": len(
                        {
                            (item.specimen_id, item.run_id)
                            for item in observations
                            if item.material == material
                            and item.morphology == morphology
                        }
                    ),
                    "specimens": len(
                        {
                            item.specimen_id
                            for item in observations
                            if item.material == material
                            and item.morphology == morphology
                        }
                    ),
                }
                for material in config.materials
                for morphology in config.morphologies
            ],
        },
        "force_summary_n": {
            "minimum": min(item.actual_force_min_n for item in observations),
            "median": float(np.median([item.actual_force_median_n for item in observations])),
            "maximum": max(item.actual_force_max_n for item in observations),
        },
        "split_definition": split_manifest | {"split_records": "see fig6c_split_manifest.json"},
        "validity_rules": {
            "geometry": "five unloaded-image lattice teeth pass fixed contrast QC",
            "contact_valid": "thresholded unloaded-noise evidence, peak SNR, and saturation QC",
            "geometry_localization_valid": "unchanged production centroid spatial checks and physical mapping",
            "template_localization_valid": "contact-valid finite registered response and complete selected templates",
            "k1_to_k4_calibration": "all selected calibration episodes contact-valid; no budget replacement",
        },
        "aggregation_rules": {
            "panel_a": "run first",
            "panel_b": "ratio of aggregate D and aggregate W",
            "panel_c": "calibration subsets averaged within outer repetition, then outer repetitions summarized",
        },
        "metric_units": {
            "W_cycle": "DN",
            "W_recontact": "DN/N",
            "D_neighbor": "DN/N",
            "Q_recontact": "dimensionless",
            "localization_mae": "mm",
        },
        "missing_conditions": [
            {
                "panel": "a",
                "condition": "30 mm maintained-contact",
                "reason": "contact-history sessions contain only sphere_10mm",
            }
        ],
        "exclusions": [],
        "fig5_outputs_modified": False,
        "fig6d_recomputed": False,
        "fig6e_replayed": False,
    }
    provenance_path = output / "fig6c_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    paths["provenance"] = provenance_path
    return paths


def cached_artifacts_current(
    config: PaperFigureConfig,
    config_path: Path,
    *,
    artifact_path: Path = DEFAULT_CONTACT_DATASET_H5,
) -> bool:
    """Return whether the versioned output fingerprint matches current inputs."""

    output = config.analysis_output_directory / OUTPUT_SUBDIRECTORY
    provenance_path = output / "fig6c_provenance.json"
    required = (
        provenance_path,
        output / "fig6a_contact_state_variability.csv",
        output / "fig6b_distinguishability_values.csv",
        output / "fig6c_contact_observations.npz",
        output / "fig6c_geometry_anchors.csv",
        output / "fig6c_split_manifest.json",
        output / "fig6c_predictions.csv",
        output / "fig6c_summary.csv",
    )
    if any(not path.is_file() for path in required):
        return False
    analysis_inputs = [
        HISTORY_ROOTS[material] / "results" / "same_contact_repeatability.csv"
        for material in config.materials
    ]
    analysis_inputs.extend(
        _morphology_metric_path(config, material, morphology, indenter)
        for material in config.materials
        for indenter in config.indenters
        for morphology in config.morphologies
    )
    raw_digest, _ = _relevant_raw_input_digest(artifact_path)
    fingerprint, _ = _protocol_fingerprint(
        config_path, artifact_path, analysis_inputs, raw_digest
    )
    stored = json.loads(provenance_path.read_text(encoding="utf-8"))
    return stored.get("protocol_fingerprint") == fingerprint


__all__ = [
    "OUTPUT_SUBDIRECTORY",
    "PROTOCOL_ID",
    "aggregate_run_first_cycle_values",
    "aggregate_panel_a",
    "cached_artifacts_current",
    "detect_led_anchor_lattice",
    "enumerate_repetition_splits",
    "evaluate_optional_calibration",
    "panel_b_values",
    "physical_coordinates_from_led_rows",
    "run_analysis",
    "summarize_optional_calibration_predictions",
    "validated_recontact_ratio",
]

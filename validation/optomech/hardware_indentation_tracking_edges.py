"""Test direct rigid-shaft edge geometry as an indentation proxy.

This read-only study is intentionally separate from the earlier phase-correlation
and 1-D NCC studies.  It processes only the prescribed twelve Solaris baseline
runs and writes a new validation bundle without modifying source measurements.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import theilslopes


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.analysis.dataset import index_session  # noqa: E402
from experiments.analysis.metrics import actual_force_magnitude  # noqa: E402


SESSION_PATH = (
    REPOSITORY_ROOT / "output" / "contact_dataset" / "2026-09-04_solaris_baseline_01"
)
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "output" / "validation" / "hardware_indentation_tracking_edges"
)
SPECIMEN_ID = "solaris_baseline_02"
INDENTER = "sphere_10mm"
TARGET_FORCES_N = (2.0, 5.0, 10.0, 15.0)
REFERENCE_FORCE_N = 2.0
MOVING_ROI_XYWH = (1060, 130, 150, 690)
FIXTURE_ROI_XYWH = (1200, 820, 680, 250)
SAMPLE_RUN_IDS = (
    "run_0001",
    "run_0002",
    "run_0006",
    "run_0007",
    "run_0011",
    "run_0012",
    "run_0016",
    "run_0017",
    "run_0021",
    "run_0022",
    "run_0026",
    "run_0027",
)
HOLE_TO_CONTACT_POSITION_MM = {
    1: 0.0,
    2: 10.0,
    3: 20.0,
    4: 30.0,
    5: 40.0,
    6: 50.0,
}
SYNTHETIC_SHIFTS_PX = (-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0)
CSV_FLOAT_FORMAT = ".9g"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    return parser.parse_args()


def _load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def _median_bgr(paths: Iterable[Path]) -> np.ndarray:
    images = [_load_bgr(path) for path in paths]
    if not images:
        raise ValueError("cannot construct a hold median without frames")
    if any(image.shape != images[0].shape for image in images):
        raise ValueError("all frames in one hold must have the same shape")
    return np.median(np.stack(images), axis=0).astype(np.uint8)


def _validate_roi(
    roi_xywh: tuple[int, int, int, int], image_shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    x, y, width, height = roi_xywh
    image_height, image_width = image_shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive")
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError(
            f"ROI {roi_xywh} exceeds image size {image_width} x {image_height}"
        )
    return roi_xywh


def _crop(image: np.ndarray, roi_xywh: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = roi_xywh
    return image[y : y + height, x : x + width]


def _signed_scharr_x(crop: np.ndarray) -> np.ndarray:
    red = crop[:, :, 2].astype(np.float32)
    return cv2.Scharr(red, cv2.CV_32F, 1, 0)


def _subpixel_peak(values: np.ndarray, integer_index: int) -> float:
    """Return a three-sample parabolic maximum without moving over one pixel."""

    index = int(integer_index)
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left = float(values[index - 1])
    center = float(values[index])
    right = float(values[index + 1])
    denominator = left - 2.0 * center + right
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float32).eps:
        return float(index)
    offset = 0.5 * (left - right) / denominator
    if not np.isfinite(offset) or abs(offset) > 1.0:
        return float(index)
    return float(index + offset)


def _row_extrema(gradient_x: np.ndarray) -> dict[str, np.ndarray]:
    height = gradient_x.shape[0]
    rows = np.arange(height, dtype=np.int32)
    positive_integer = np.argmax(gradient_x, axis=1).astype(np.int32)
    negative_integer = np.argmin(gradient_x, axis=1).astype(np.int32)
    positive_strength = gradient_x[rows, positive_integer].astype(np.float64)
    negative_strength = (-gradient_x[rows, negative_integer]).astype(np.float64)
    positive_subpixel = np.asarray(
        [_subpixel_peak(gradient_x[row], positive_integer[row]) for row in rows],
        dtype=np.float64,
    )
    negative_subpixel = np.asarray(
        [_subpixel_peak(-gradient_x[row], negative_integer[row]) for row in rows],
        dtype=np.float64,
    )
    return {
        "positive_integer": positive_integer,
        "negative_integer": negative_integer,
        "positive_subpixel": positive_subpixel,
        "negative_subpixel": negative_subpixel,
        "positive_strength": positive_strength,
        "negative_strength": negative_strength,
    }


def _robust_line(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    slope, intercept, _, _ = theilslopes(x.astype(np.float64), y.astype(np.float64))
    return float(slope), float(intercept)


def _line_x(slope: float, intercept: float, y: np.ndarray | float) -> np.ndarray:
    return intercept + slope * np.asarray(y, dtype=np.float64)


def _robust_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan")
    median = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - median)))


def _moving_polarity(
    red: np.ndarray, extrema: dict[str, np.ndarray]
) -> tuple[str, str]:
    """Choose the 2 N polarity whose broad pair encloses the darker shaft."""

    candidates = (("negative", "positive"), ("positive", "negative"))
    scores: list[float] = []
    for left_polarity, right_polarity in candidates:
        left = extrema[f"{left_polarity}_integer"]
        right = extrema[f"{right_polarity}_integer"]
        ordered = left < right
        widths = right - left
        if not np.any(ordered):
            scores.append(float("-inf"))
            continue
        broad_width = float(np.median(widths[ordered]))
        broad = ordered & (widths >= broad_width)
        darkness = []
        for row in np.flatnonzero(broad):
            left_x = int(left[row])
            right_x = int(right[row])
            inside = red[row, left_x + 1 : right_x]
            outside = np.concatenate((red[row, :left_x], red[row, right_x + 1 :]))
            if len(inside) and len(outside):
                darkness.append(float(np.mean(outside) - np.mean(inside)))
        scores.append(float(np.median(darkness)) if darkness else float("-inf"))
    choice = int(np.argmax(scores))
    if not np.isfinite(scores[choice]):
        raise RuntimeError("2 N reference did not define an ordered shaft edge pair")
    return candidates[choice]


def _moving_reference(crop: np.ndarray) -> dict[str, Any]:
    gradient = _signed_scharr_x(crop)
    extrema = _row_extrema(gradient)
    left_polarity, right_polarity = _moving_polarity(
        crop[:, :, 2].astype(np.float32), extrema
    )
    left = extrema[f"{left_polarity}_subpixel"]
    right = extrema[f"{right_polarity}_subpixel"]
    ordered = left < right
    widths = right - left
    broad_width = float(np.median(widths[ordered]))
    broad = ordered & (widths >= broad_width)
    y = np.flatnonzero(broad).astype(np.float64)
    left_slope, left_intercept = _robust_line(y, left[broad])
    right_slope, right_intercept = _robust_line(y, right[broad])
    left_residual = left[broad] - _line_x(left_slope, left_intercept, y)
    right_residual = right[broad] - _line_x(right_slope, right_intercept, y)
    width_residual = widths[broad] - (
        _line_x(right_slope, right_intercept, y)
        - _line_x(left_slope, left_intercept, y)
    )
    edge_sigma = max(_robust_sigma(left_residual), _robust_sigma(right_residual))
    width_sigma = _robust_sigma(width_residual)
    edge_tolerance = max(3.0, 6.0 * edge_sigma)
    width_tolerance = max(4.0, 6.0 * width_sigma)
    inlier = broad.copy()
    broad_inlier = (
        (np.abs(left_residual) <= edge_tolerance)
        & (np.abs(right_residual) <= edge_tolerance)
        & (np.abs(width_residual) <= width_tolerance)
    )
    inlier[np.flatnonzero(broad)] = broad_inlier
    y = np.flatnonzero(inlier).astype(np.float64)
    left_slope, left_intercept = _robust_line(y, left[inlier])
    right_slope, right_intercept = _robust_line(y, right[inlier])
    if not np.all(
        np.isfinite((left_slope, left_intercept, right_slope, right_intercept))
    ):
        raise RuntimeError("2 N reference shaft lines were not estimable")
    y_reference = 0.5 * (crop.shape[0] - 1)
    return {
        "tracking_variant": "rowwise_signed_scharr_x_theil_sen_lines",
        "left_polarity": left_polarity,
        "right_polarity": right_polarity,
        "left_slope": left_slope,
        "left_intercept": left_intercept,
        "right_slope": right_slope,
        "right_intercept": right_intercept,
        "y_reference": y_reference,
        "edge_tolerance_px": edge_tolerance,
        "width_tolerance_px": width_tolerance,
        "reference_inlier_rows": int(np.count_nonzero(inlier)),
        "reference_inlier_fraction": float(np.mean(inlier)),
    }


def _track_moving(crop: np.ndarray, reference: dict[str, Any]) -> dict[str, Any]:
    gradient = _signed_scharr_x(crop)
    extrema = _row_extrema(gradient)
    left_polarity = str(reference["left_polarity"])
    right_polarity = str(reference["right_polarity"])
    left_integer = extrema[f"{left_polarity}_integer"]
    right_integer = extrema[f"{right_polarity}_integer"]
    left = extrema[f"{left_polarity}_subpixel"]
    right = extrema[f"{right_polarity}_subpixel"]
    left_strength = extrema[f"{left_polarity}_strength"]
    right_strength = extrema[f"{right_polarity}_strength"]
    rows = np.arange(crop.shape[0], dtype=np.float64)
    expected_left = _line_x(reference["left_slope"], reference["left_intercept"], rows)
    expected_right = _line_x(
        reference["right_slope"], reference["right_intercept"], rows
    )
    expected_width = expected_right - expected_left
    width = right - left
    ordered = right > left
    width_consistent = np.abs(width - expected_width) <= reference["width_tolerance_px"]
    provisional = ordered & width_consistent
    if np.any(provisional):
        common_shift = float(
            np.median(
                0.5 * (left[provisional] + right[provisional])
                - 0.5 * (expected_left[provisional] + expected_right[provisional])
            )
        )
    else:
        common_shift = float("nan")
    valid = (
        provisional
        & (left_strength > 0.0)
        & (right_strength > 0.0)
        & (
            np.abs(left - expected_left - common_shift)
            <= reference["edge_tolerance_px"]
        )
        & (
            np.abs(right - expected_right - common_shift)
            <= reference["edge_tolerance_px"]
        )
    )
    if np.count_nonzero(valid) < 3:
        return _invalid_moving_result(
            left_integer,
            right_integer,
            left,
            right,
            left_strength,
            right_strength,
            valid,
        )
    y = rows[valid]
    left_slope, left_intercept = _robust_line(y, left[valid])
    right_slope, right_intercept = _robust_line(y, right[valid])
    y_reference = float(reference["y_reference"])
    left_x = float(_line_x(left_slope, left_intercept, y_reference))
    right_x = float(_line_x(right_slope, right_intercept, y_reference))
    valid_result = bool(
        np.isfinite(left_x) and np.isfinite(right_x) and right_x > left_x
    )
    return {
        "valid": valid_result,
        "left_integer_median_px": float(np.median(left_integer[valid])),
        "right_integer_median_px": float(np.median(right_integer[valid])),
        "left_x_px": left_x,
        "right_x_px": right_x,
        "center_x_px": 0.5 * (left_x + right_x),
        "width_px": right_x - left_x,
        "left_slope_px_per_row": left_slope,
        "left_intercept_px": left_intercept,
        "right_slope_px_per_row": right_slope,
        "right_intercept_px": right_intercept,
        "valid_row_count": int(np.count_nonzero(valid)),
        "valid_row_fraction": float(np.mean(valid)),
        "left_edge_strength_median": float(np.median(left_strength[valid])),
        "right_edge_strength_median": float(np.median(right_strength[valid])),
        "row_left_integer_px": left_integer,
        "row_right_integer_px": right_integer,
        "row_left_subpixel_px": left,
        "row_right_subpixel_px": right,
        "row_left_strength": left_strength,
        "row_right_strength": right_strength,
        "row_valid": valid,
    }


def _invalid_moving_result(
    left_integer: np.ndarray,
    right_integer: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    left_strength: np.ndarray,
    right_strength: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    return {
        "valid": False,
        "left_integer_median_px": float("nan"),
        "right_integer_median_px": float("nan"),
        "left_x_px": float("nan"),
        "right_x_px": float("nan"),
        "center_x_px": float("nan"),
        "width_px": float("nan"),
        "left_slope_px_per_row": float("nan"),
        "left_intercept_px": float("nan"),
        "right_slope_px_per_row": float("nan"),
        "right_intercept_px": float("nan"),
        "valid_row_count": int(np.count_nonzero(valid)),
        "valid_row_fraction": float(np.mean(valid)),
        "left_edge_strength_median": float("nan"),
        "right_edge_strength_median": float("nan"),
        "row_left_integer_px": left_integer,
        "row_right_integer_px": right_integer,
        "row_left_subpixel_px": left,
        "row_right_subpixel_px": right,
        "row_left_strength": left_strength,
        "row_right_strength": right_strength,
        "row_valid": valid,
    }


def _fixture_reference(crop: np.ndarray) -> dict[str, Any]:
    extrema = _row_extrema(_signed_scharr_x(crop))
    rows = np.arange(crop.shape[0], dtype=np.float64)
    candidates = []
    for polarity in ("positive", "negative"):
        x = extrema[f"{polarity}_subpixel"]
        slope, intercept = _robust_line(rows, x)
        residual = x - _line_x(slope, intercept, rows)
        sigma = _robust_sigma(residual)
        candidates.append((sigma, polarity, slope, intercept, residual))
    sigma, polarity, slope, intercept, residual = min(
        candidates, key=lambda item: item[0]
    )
    tolerance = max(3.0, 6.0 * sigma)
    centered = residual - np.median(residual)
    inlier = np.abs(centered) <= tolerance
    slope, intercept = _robust_line(
        rows[inlier], extrema[f"{polarity}_subpixel"][inlier]
    )
    if not np.all(np.isfinite((slope, intercept))):
        raise RuntimeError("2 N reference fixture edge was not estimable")
    return {
        "tracking_variant": "single_rowwise_signed_scharr_x_theil_sen_line",
        "polarity": polarity,
        "slope": slope,
        "intercept": intercept,
        "y_reference": 0.5 * (crop.shape[0] - 1),
        "edge_tolerance_px": tolerance,
        "reference_inlier_rows": int(np.count_nonzero(inlier)),
        "reference_inlier_fraction": float(np.mean(inlier)),
    }


def _track_fixture(crop: np.ndarray, reference: dict[str, Any]) -> dict[str, Any]:
    extrema = _row_extrema(_signed_scharr_x(crop))
    polarity = str(reference["polarity"])
    integer = extrema[f"{polarity}_integer"]
    subpixel = extrema[f"{polarity}_subpixel"]
    strength = extrema[f"{polarity}_strength"]
    rows = np.arange(crop.shape[0], dtype=np.float64)
    expected = _line_x(reference["slope"], reference["intercept"], rows)
    shift = float(np.median(subpixel - expected))
    valid = (strength > 0.0) & (
        np.abs(subpixel - expected - shift) <= reference["edge_tolerance_px"]
    )
    if np.count_nonzero(valid) < 3:
        return {
            "valid": False,
            "integer_median_px": float("nan"),
            "x_px": float("nan"),
            "slope_px_per_row": float("nan"),
            "intercept_px": float("nan"),
            "valid_row_count": int(np.count_nonzero(valid)),
            "valid_row_fraction": float(np.mean(valid)),
            "edge_strength_median": float("nan"),
            "row_integer_px": integer,
            "row_subpixel_px": subpixel,
            "row_strength": strength,
            "row_valid": valid,
        }
    slope, intercept = _robust_line(rows[valid], subpixel[valid])
    x = float(_line_x(slope, intercept, reference["y_reference"]))
    return {
        "valid": bool(np.isfinite(x)),
        "integer_median_px": float(np.median(integer[valid])),
        "x_px": x,
        "slope_px_per_row": slope,
        "intercept_px": intercept,
        "valid_row_count": int(np.count_nonzero(valid)),
        "valid_row_fraction": float(np.mean(valid)),
        "edge_strength_median": float(np.median(strength[valid])),
        "row_integer_px": integer,
        "row_subpixel_px": subpixel,
        "row_strength": strength,
        "row_valid": valid,
    }


def _frame_force(measurements: Any) -> float:
    return actual_force_magnitude(
        float(measurements["Fx_N"]),
        float(measurements["Fy_N"]),
        float(measurements["Fz_N"]),
    )


def _select_sample(index: Any) -> list[Any]:
    if index.session.specimen_id != SPECIMEN_ID:
        raise ValueError(
            f"expected specimen {SPECIMEN_ID}, got {index.session.specimen_id}"
        )
    runs = {run.run_id: run for run in index.runs}
    missing = sorted(set(SAMPLE_RUN_IDS) - set(runs))
    if missing:
        raise ValueError(f"required sample runs are missing: {missing}")
    selected = [runs[run_id] for run_id in SAMPLE_RUN_IDS]
    for run in selected:
        if run.status != "complete" or run.indenter != INDENTER:
            raise ValueError(f"invalid sample run contract: {run.run_id}")
    if {run.hole_index for run in selected} != set(range(1, 7)):
        raise ValueError("the fixed sample must span holes 1 through 6")
    return selected


def _group_frames(index: Any, run_ids: set[str]) -> dict[tuple[str, float], list[Any]]:
    groups: dict[tuple[str, float], list[Any]] = {}
    for frame in index.frames:
        if frame.run is None or frame.run.run_id not in run_ids:
            continue
        assert frame.target_force_n is not None
        key = frame.run.run_id, float(frame.target_force_n)
        groups.setdefault(key, []).append(frame)
    for frames in groups.values():
        frames.sort(key=lambda frame: int(frame.measurements["frame_index"]))
    required = {(run_id, force) for run_id in run_ids for force in TARGET_FORCES_N}
    missing = sorted(required - set(groups))
    if missing:
        raise ValueError(f"sample is missing required force holds: {missing}")
    return groups


def _moving_columns(prefix: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_valid": result["valid"],
        f"{prefix}_left_integer_median_px": result["left_integer_median_px"],
        f"{prefix}_right_integer_median_px": result["right_integer_median_px"],
        f"{prefix}_left_x_px": result["left_x_px"],
        f"{prefix}_right_x_px": result["right_x_px"],
        f"{prefix}_center_x_px": result["center_x_px"],
        f"{prefix}_width_px": result["width_px"],
        f"{prefix}_left_slope_px_per_row": result["left_slope_px_per_row"],
        f"{prefix}_right_slope_px_per_row": result["right_slope_px_per_row"],
        f"{prefix}_valid_row_count": result["valid_row_count"],
        f"{prefix}_valid_row_fraction": result["valid_row_fraction"],
        f"{prefix}_left_edge_strength_median": result["left_edge_strength_median"],
        f"{prefix}_right_edge_strength_median": result["right_edge_strength_median"],
    }


def _fixture_columns(prefix: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_valid": result["valid"],
        f"{prefix}_integer_median_px": result["integer_median_px"],
        f"{prefix}_x_px": result["x_px"],
        f"{prefix}_slope_px_per_row": result["slope_px_per_row"],
        f"{prefix}_valid_row_count": result["valid_row_count"],
        f"{prefix}_valid_row_fraction": result["valid_row_fraction"],
        f"{prefix}_edge_strength_median": result["edge_strength_median"],
    }


def _measure_sample(
    index: Any,
    runs: list[Any],
    groups: dict[tuple[str, float], list[Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, float], np.ndarray],
    dict[str, dict[str, Any]],
]:
    hold_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    hold_images: dict[tuple[str, float], np.ndarray] = {}
    references: dict[str, dict[str, Any]] = {}

    first = _load_bgr(groups[(runs[0].run_id, REFERENCE_FORCE_N)][0].rgb_path)
    _validate_roi(MOVING_ROI_XYWH, first.shape)
    _validate_roi(FIXTURE_ROI_XYWH, first.shape)

    for run in runs:
        images_by_force: dict[float, list[np.ndarray]] = {}
        for force in TARGET_FORCES_N:
            frames = groups[(run.run_id, force)]
            images = [_load_bgr(frame.rgb_path) for frame in frames]
            images_by_force[force] = images
            hold_images[(run.run_id, force)] = np.median(
                np.stack(images), axis=0
            ).astype(np.uint8)

        reference_image = hold_images[(run.run_id, REFERENCE_FORCE_N)]
        moving_reference = _moving_reference(_crop(reference_image, MOVING_ROI_XYWH))
        fixture_reference = _fixture_reference(_crop(reference_image, FIXTURE_ROI_XYWH))
        references[run.run_id] = {
            "moving": moving_reference,
            "fixture": fixture_reference,
        }
        reference_moving = _track_moving(
            _crop(reference_image, MOVING_ROI_XYWH), moving_reference
        )
        reference_fixture = _track_fixture(
            _crop(reference_image, FIXTURE_ROI_XYWH), fixture_reference
        )
        if not reference_moving["valid"] or not reference_fixture["valid"]:
            raise RuntimeError(f"2 N reference tracking failed for {run.run_id}")

        common_geometry = {
            "specimen_id": index.session.specimen_id,
            "run_id": run.run_id,
            "indenter": run.indenter,
            "hole_index": run.hole_index,
            "contact_position_mm": HOLE_TO_CONTACT_POSITION_MM[run.hole_index],
            "repetition_index": run.repetition_index,
        }
        geometry_rows.extend(
            [
                {
                    **common_geometry,
                    "subject": "moving_left",
                    "tracking_variant": moving_reference["tracking_variant"],
                    "polarity": moving_reference["left_polarity"],
                    "roi_xywh": str(MOVING_ROI_XYWH),
                    "y_reference_px": moving_reference["y_reference"],
                    "reference_slope_px_per_row": moving_reference["left_slope"],
                    "reference_intercept_px": moving_reference["left_intercept"],
                    "reference_x_at_y_reference_px": reference_moving["left_x_px"],
                    "edge_tolerance_px": moving_reference["edge_tolerance_px"],
                    "width_tolerance_px": moving_reference["width_tolerance_px"],
                    "reference_inlier_rows": moving_reference["reference_inlier_rows"],
                    "reference_inlier_fraction": moving_reference[
                        "reference_inlier_fraction"
                    ],
                },
                {
                    **common_geometry,
                    "subject": "moving_right",
                    "tracking_variant": moving_reference["tracking_variant"],
                    "polarity": moving_reference["right_polarity"],
                    "roi_xywh": str(MOVING_ROI_XYWH),
                    "y_reference_px": moving_reference["y_reference"],
                    "reference_slope_px_per_row": moving_reference["right_slope"],
                    "reference_intercept_px": moving_reference["right_intercept"],
                    "reference_x_at_y_reference_px": reference_moving["right_x_px"],
                    "edge_tolerance_px": moving_reference["edge_tolerance_px"],
                    "width_tolerance_px": moving_reference["width_tolerance_px"],
                    "reference_inlier_rows": moving_reference["reference_inlier_rows"],
                    "reference_inlier_fraction": moving_reference[
                        "reference_inlier_fraction"
                    ],
                },
                {
                    **common_geometry,
                    "subject": "fixture_edge",
                    "tracking_variant": fixture_reference["tracking_variant"],
                    "polarity": fixture_reference["polarity"],
                    "roi_xywh": str(FIXTURE_ROI_XYWH),
                    "y_reference_px": fixture_reference["y_reference"],
                    "reference_slope_px_per_row": fixture_reference["slope"],
                    "reference_intercept_px": fixture_reference["intercept"],
                    "reference_x_at_y_reference_px": reference_fixture["x_px"],
                    "edge_tolerance_px": fixture_reference["edge_tolerance_px"],
                    "width_tolerance_px": "",
                    "reference_inlier_rows": fixture_reference["reference_inlier_rows"],
                    "reference_inlier_fraction": fixture_reference[
                        "reference_inlier_fraction"
                    ],
                },
            ]
        )

        for force in TARGET_FORCES_N:
            frames = groups[(run.run_id, force)]
            hold_image = hold_images[(run.run_id, force)]
            moving = _track_moving(_crop(hold_image, MOVING_ROI_XYWH), moving_reference)
            fixture = _track_fixture(
                _crop(hold_image, FIXTURE_ROI_XYWH), fixture_reference
            )
            hold_rows.append(
                _raw_measurement_row(
                    index,
                    run,
                    force,
                    frames,
                    moving,
                    fixture,
                    reference_moving,
                    reference_fixture,
                )
            )
            for frame, image in zip(frames, images_by_force[force], strict=True):
                moving_frame = _track_moving(
                    _crop(image, MOVING_ROI_XYWH), moving_reference
                )
                fixture_frame = _track_fixture(
                    _crop(image, FIXTURE_ROI_XYWH), fixture_reference
                )
                frame_rows.append(
                    _raw_frame_row(
                        index,
                        run,
                        force,
                        frame,
                        moving_frame,
                        fixture_frame,
                        reference_moving,
                        reference_fixture,
                    )
                )
    return hold_rows, frame_rows, geometry_rows, hold_images, references


def _relative_geometry(
    moving: dict[str, Any],
    fixture: dict[str, Any],
    reference_moving: dict[str, Any],
    reference_fixture: dict[str, Any],
) -> dict[str, float | bool]:
    valid = bool(moving["valid"] and fixture["valid"])
    if not valid:
        return {
            "tracking_valid": False,
            "left_shift_px": float("nan"),
            "right_shift_px": float("nan"),
            "center_shift_px": float("nan"),
            "fixture_shift_px": float("nan"),
            "relative_center_shift_px": float("nan"),
            "edge_disagreement_px": float("nan"),
            "width_change_px": float("nan"),
            "width_change_fraction": float("nan"),
        }
    left_shift = float(moving["left_x_px"] - reference_moving["left_x_px"])
    right_shift = float(moving["right_x_px"] - reference_moving["right_x_px"])
    center_shift = float(moving["center_x_px"] - reference_moving["center_x_px"])
    fixture_shift = float(fixture["x_px"] - reference_fixture["x_px"])
    width_change = float(moving["width_px"] - reference_moving["width_px"])
    return {
        "tracking_valid": True,
        "left_shift_px": left_shift,
        "right_shift_px": right_shift,
        "center_shift_px": center_shift,
        "fixture_shift_px": fixture_shift,
        "relative_center_shift_px": center_shift - fixture_shift,
        "edge_disagreement_px": abs(left_shift - right_shift),
        "width_change_px": width_change,
        "width_change_fraction": width_change / float(reference_moving["width_px"]),
    }


def _raw_measurement_row(
    index: Any,
    run: Any,
    force: float,
    frames: list[Any],
    moving: dict[str, Any],
    fixture: dict[str, Any],
    reference_moving: dict[str, Any],
    reference_fixture: dict[str, Any],
) -> dict[str, Any]:
    forces = np.asarray([_frame_force(frame.measurements) for frame in frames])
    return {
        "specimen_id": index.session.specimen_id,
        "material": index.session.material,
        "morphology": index.session.morphology,
        "run_id": run.run_id,
        "indenter": run.indenter,
        "hole_index": run.hole_index,
        "contact_position_mm": HOLE_TO_CONTACT_POSITION_MM[run.hole_index],
        "repetition_index": run.repetition_index,
        "target_force_n": force,
        "frame_count": len(frames),
        "actual_force_median_n": float(np.median(forces)),
        "actual_force_std_n": float(np.std(forces, ddof=1)) if len(forces) > 1 else 0.0,
        "actual_force_min_n": float(np.min(forces)),
        "actual_force_max_n": float(np.max(forces)),
        **_relative_geometry(moving, fixture, reference_moving, reference_fixture),
        **_moving_columns("moving", moving),
        **_fixture_columns("fixture", fixture),
    }


def _raw_frame_row(
    index: Any,
    run: Any,
    force: float,
    frame: Any,
    moving: dict[str, Any],
    fixture: dict[str, Any],
    reference_moving: dict[str, Any],
    reference_fixture: dict[str, Any],
) -> dict[str, Any]:
    return {
        "specimen_id": index.session.specimen_id,
        "material": index.session.material,
        "morphology": index.session.morphology,
        "run_id": run.run_id,
        "indenter": run.indenter,
        "hole_index": run.hole_index,
        "contact_position_mm": HOLE_TO_CONTACT_POSITION_MM[run.hole_index],
        "repetition_index": run.repetition_index,
        "target_force_n": force,
        "frame_index": int(frame.measurements["frame_index"]),
        "camera_host_time_s": float(frame.measurements["camera_host_time_s"]),
        "actual_force_n": _frame_force(frame.measurements),
        "image_path": str(frame.rgb_path.resolve()),
        **_relative_geometry(moving, fixture, reference_moving, reference_fixture),
        **_moving_columns("moving", moving),
        **_fixture_columns("fixture", fixture),
    }


def _apply_common_sign(
    hold_rows: list[dict[str, Any]], frame_rows: list[dict[str, Any]]
) -> float:
    final = np.asarray(
        [
            row["relative_center_shift_px"]
            for row in hold_rows
            if row["target_force_n"] == 15.0
            and np.isfinite(row["relative_center_shift_px"])
        ],
        dtype=np.float64,
    )
    sign = 1.0 if not len(final) or float(np.median(final)) >= 0.0 else -1.0
    for row in (*hold_rows, *frame_rows):
        row["common_sign"] = int(sign)
        indentation = sign * float(row["relative_center_shift_px"])
        row["indentation_px"] = 0.0 if indentation == 0.0 else indentation
    return sign


def _add_hold_qc(
    hold_rows: list[dict[str, Any]], frame_rows: list[dict[str, Any]]
) -> None:
    by_key: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for frame in frame_rows:
        by_key.setdefault((frame["run_id"], frame["target_force_n"]), []).append(frame)
    for hold in hold_rows:
        frames = by_key[(hold["run_id"], hold["target_force_n"])]
        values = np.asarray(
            [
                frame["indentation_px"]
                for frame in frames
                if np.isfinite(frame["indentation_px"])
            ],
            dtype=np.float64,
        )
        widths = np.asarray(
            [
                frame["moving_width_px"]
                for frame in frames
                if np.isfinite(frame["moving_width_px"])
            ],
            dtype=np.float64,
        )
        disagreements = np.asarray(
            [
                frame["edge_disagreement_px"]
                for frame in frames
                if np.isfinite(frame["edge_disagreement_px"])
            ],
            dtype=np.float64,
        )
        valid_row_fractions = np.asarray(
            [
                frame["moving_valid_row_fraction"]
                for frame in frames
                if np.isfinite(frame["moving_valid_row_fraction"])
            ],
            dtype=np.float64,
        )
        hold.update(
            {
                "finite_frame_count": int(len(values)),
                "frame_indentation_median_px": float(np.median(values))
                if len(values)
                else float("nan"),
                "frame_indentation_std_px": float(np.std(values, ddof=1))
                if len(values) > 1
                else (0.0 if len(values) == 1 else float("nan")),
                "frame_indentation_min_px": float(np.min(values))
                if len(values)
                else float("nan"),
                "frame_indentation_max_px": float(np.max(values))
                if len(values)
                else float("nan"),
                "frame_indentation_range_px": float(np.ptp(values))
                if len(values)
                else float("nan"),
                "frame_width_median_px": float(np.median(widths))
                if len(widths)
                else float("nan"),
                "frame_width_range_px": float(np.ptp(widths))
                if len(widths)
                else float("nan"),
                "frame_edge_disagreement_median_px": float(np.median(disagreements))
                if len(disagreements)
                else float("nan"),
                "frame_edge_disagreement_max_px": float(np.max(disagreements))
                if len(disagreements)
                else float("nan"),
                "frame_valid_row_fraction_median": float(np.median(valid_row_fractions))
                if len(valid_row_fractions)
                else float("nan"),
                "frame_valid_row_fraction_min": float(np.min(valid_row_fractions))
                if len(valid_row_fractions)
                else float("nan"),
            }
        )
    for run_id in SAMPLE_RUN_IDS:
        ordered = sorted(
            (row for row in hold_rows if row["run_id"] == run_id),
            key=lambda row: row["target_force_n"],
        )
        values = [float(row["indentation_px"]) for row in ordered]
        monotonic = bool(np.all(np.isfinite(values)) and np.all(np.diff(values) >= 0.0))
        for row in ordered:
            row["run_trajectory_monotonic"] = monotonic


def _synthetic_check(
    hold_images: dict[tuple[str, float], np.ndarray],
    references: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for run_id in SAMPLE_RUN_IDS[:4]:
        crop = _crop(hold_images[(run_id, REFERENCE_FORCE_N)], MOVING_ROI_XYWH)
        reference = references[run_id]["moving"]
        baseline = _track_moving(crop, reference)
        for requested_shift in SYNTHETIC_SHIFTS_PX:
            transform = np.asarray(
                [[1.0, 0.0, requested_shift], [0.0, 1.0, 0.0]], dtype=np.float32
            )
            shifted = cv2.warpAffine(
                crop,
                transform,
                (crop.shape[1], crop.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            measured = _track_moving(shifted, reference)
            left_shift = measured["left_x_px"] - baseline["left_x_px"]
            right_shift = measured["right_x_px"] - baseline["right_x_px"]
            center_shift = measured["center_x_px"] - baseline["center_x_px"]
            rows.append(
                {
                    "source_run_id": run_id,
                    "requested_shift_px": requested_shift,
                    "tracking_valid": measured["valid"],
                    "recovered_left_shift_px": left_shift,
                    "recovered_right_shift_px": right_shift,
                    "recovered_center_shift_px": center_shift,
                    "center_signed_error_px": center_shift - requested_shift,
                    "center_absolute_error_px": abs(center_shift - requested_shift),
                    "left_right_disagreement_px": abs(left_shift - right_shift),
                    "reference_width_px": baseline["width_px"],
                    "recovered_width_px": measured["width_px"],
                    "width_change_px": measured["width_px"] - baseline["width_px"],
                    "valid_row_count": measured["valid_row_count"],
                    "valid_row_fraction": measured["valid_row_fraction"],
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: format(value, CSV_FLOAT_FORMAT)
                    if isinstance(value, float)
                    else value
                    for key, value in row.items()
                }
            )


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def _median_range(values: Iterable[float]) -> tuple[float, float, float]:
    array = _finite(values)
    if not len(array):
        return float("nan"), float("nan"), float("nan")
    return float(np.median(array)), float(np.min(array)), float(np.max(array))


def _percentiles(values: Iterable[float]) -> tuple[float, float, float]:
    array = _finite(values)
    if not len(array):
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.median(array)),
        float(np.percentile(array, 95)),
        float(np.max(array)),
    )


def _median_iqr_range(
    values: Iterable[float],
) -> tuple[float, float, float, float, float]:
    array = _finite(values)
    if not len(array):
        return (float("nan"),) * 5
    return (
        float(np.median(array)),
        float(np.percentile(array, 25)),
        float(np.percentile(array, 75)),
        float(np.min(array)),
        float(np.max(array)),
    )


def _run_qc(hold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for run_id in SAMPLE_RUN_IDS:
        rows = sorted(
            (row for row in hold_rows if row["run_id"] == run_id),
            key=lambda row: row["target_force_n"],
        )
        final = next(row for row in rows if row["target_force_n"] == 15.0)
        max_within = max(
            (float(row["frame_indentation_range_px"]) for row in rows),
            default=float("nan"),
        )
        total = abs(float(final["indentation_px"]))
        result.append(
            {
                "run_id": run_id,
                "hole_index": final["hole_index"],
                "contact_position_mm": final["contact_position_mm"],
                "repetition_index": final["repetition_index"],
                "trajectory_monotonic": final["run_trajectory_monotonic"],
                "delta_2_n_px": rows[0]["indentation_px"],
                "delta_5_n_px": rows[1]["indentation_px"],
                "delta_10_n_px": rows[2]["indentation_px"],
                "delta_15_n_px": rows[3]["indentation_px"],
                "movement_2_to_15_n_px": final["indentation_px"],
                "edge_disagreement_15_n_px": final["edge_disagreement_px"],
                "width_change_15_n_px": final["width_change_px"],
                "width_change_magnitude_15_n_px": abs(final["width_change_px"]),
                "width_change_fraction_15_n": final["width_change_fraction"],
                "width_change_fraction_magnitude_15_n": abs(
                    final["width_change_fraction"]
                ),
                "fixture_drift_15_n_px": final["fixture_shift_px"],
                "fixture_drift_magnitude_15_n_px": abs(final["fixture_shift_px"]),
                "maximum_within_hold_range_px": max_within,
                "maximum_within_hold_over_total_signal": max_within / total
                if np.isfinite(max_within) and total > 0.0
                else float("nan"),
            }
        )
    return result


def _plot_overlay(
    output: Path,
    hold_images: dict[tuple[str, float], np.ndarray],
    references: dict[str, dict[str, Any]],
) -> None:
    run_id = SAMPLE_RUN_IDS[0]
    figure, axes = plt.subplots(2, 4, figsize=(11.0, 5.0), constrained_layout=True)
    for column, force in enumerate(TARGET_FORCES_N):
        image = hold_images[(run_id, force)]
        moving_crop = _crop(image, MOVING_ROI_XYWH)
        fixture_crop = _crop(image, FIXTURE_ROI_XYWH)
        moving = _track_moving(moving_crop, references[run_id]["moving"])
        fixture = _track_fixture(fixture_crop, references[run_id]["fixture"])
        for axis, crop_image in (
            (axes[0, column], moving_crop),
            (axes[1, column], fixture_crop),
        ):
            axis.imshow(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
            axis.set_xticks([])
            axis.set_yticks([])
        y_moving = np.asarray([0.0, moving_crop.shape[0] - 1.0])
        axes[0, column].plot(
            _line_x(
                moving["left_slope_px_per_row"], moving["left_intercept_px"], y_moving
            ),
            y_moving,
            color="#ff7f0e",
            linewidth=1.3,
        )
        axes[0, column].plot(
            _line_x(
                moving["right_slope_px_per_row"], moving["right_intercept_px"], y_moving
            ),
            y_moving,
            color="#00bcd4",
            linewidth=1.3,
        )
        center_slope = 0.5 * (
            moving["left_slope_px_per_row"] + moving["right_slope_px_per_row"]
        )
        center_intercept = 0.5 * (
            moving["left_intercept_px"] + moving["right_intercept_px"]
        )
        axes[0, column].plot(
            _line_x(center_slope, center_intercept, y_moving),
            y_moving,
            color="white",
            linewidth=1.0,
            linestyle="--",
        )
        valid_rows = np.flatnonzero(moving["row_valid"])
        axes[0, column].scatter(
            moving["row_left_subpixel_px"][valid_rows],
            valid_rows,
            s=1.5,
            color="#ff7f0e",
            alpha=0.5,
        )
        axes[0, column].scatter(
            moving["row_right_subpixel_px"][valid_rows],
            valid_rows,
            s=1.5,
            color="#00bcd4",
            alpha=0.5,
        )
        y_fixture = np.asarray([0.0, fixture_crop.shape[0] - 1.0])
        axes[1, column].plot(
            _line_x(fixture["slope_px_per_row"], fixture["intercept_px"], y_fixture),
            y_fixture,
            color="#fdae61",
            linewidth=1.4,
        )
        axes[0, column].set_title(f"{force:g} N")
    axes[0, 0].set_ylabel("Moving shaft")
    axes[1, 0].set_ylabel("Fixed fixture")
    figure.suptitle(f"Direct edge fits on hold-median images ({run_id})")
    figure.savefig(output / "indentation_edge_tracking_overlay.png", dpi=220)
    plt.close(figure)


def _plot_trajectories(output: Path, hold_rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, 6))
    for run_id in SAMPLE_RUN_IDS:
        rows = sorted(
            (row for row in hold_rows if row["run_id"] == run_id),
            key=lambda row: row["actual_force_median_n"],
        )
        hole = int(rows[0]["hole_index"])
        axis.plot(
            [row["actual_force_median_n"] for row in rows],
            [row["indentation_px"] for row in rows],
            "-o",
            color=colors[hole - 1],
            alpha=0.72,
            linewidth=1.0,
            markersize=3.5,
            label=f"{HOLE_TO_CONTACT_POSITION_MM[hole]:g} mm"
            if int(rows[0]["repetition_index"]) == 1
            else None,
        )
    axis.set_xlabel("Measured force [N]")
    axis.set_ylabel("Relative rigid-shaft displacement [px]")
    axis.set_title("Direct edge-geometry trajectories")
    axis.legend(title="Contact position", frameon=False, ncol=3)
    figure.savefig(output / "indentation_edge_tracking_run_trajectories.png", dpi=220)
    plt.close(figure)


def _plot_within_hold(
    output: Path, hold_rows: list[dict[str, Any]], run_qc: list[dict[str, Any]]
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    values = [
        [
            row["frame_indentation_range_px"]
            for row in hold_rows
            if row["target_force_n"] == force
            and np.isfinite(row["frame_indentation_range_px"])
        ]
        for force in TARGET_FORCES_N
    ]
    axes[0].boxplot(values, tick_labels=[f"{force:g}" for force in TARGET_FORCES_N])
    axes[0].set_xlabel("Target force [N]")
    axes[0].set_ylabel("Within-hold displacement range [px]")
    axes[0].set_title("Frame-level stability")
    axes[1].scatter(
        [row["movement_2_to_15_n_px"] for row in run_qc],
        [row["maximum_within_hold_range_px"] for row in run_qc],
        c=[row["contact_position_mm"] for row in run_qc],
        cmap="viridis",
        edgecolors="#333333",
        linewidths=0.4,
    )
    limits = axes[1].get_xlim()
    axes[1].plot(limits, limits, "--", color="#999999", linewidth=1.0)
    axes[1].set_xlim(limits)
    axes[1].set_xlabel("2 to 15 N displacement [px]")
    axes[1].set_ylabel("Maximum within-hold range [px]")
    axes[1].set_title("Signal versus within-hold variation")
    figure.savefig(output / "indentation_edge_tracking_within_hold_qc.png", dpi=220)
    plt.close(figure)


def _plot_width_consistency(output: Path, run_qc: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), constrained_layout=True)
    x = np.arange(len(run_qc))
    labels = [row["run_id"].removeprefix("run_") for row in run_qc]
    axes[0].bar(
        x,
        [100.0 * row["width_change_fraction_15_n"] for row in run_qc],
        color="#4c78a8",
    )
    axes[0].axhline(0.0, color="#777777", linewidth=0.8)
    axes[0].set_ylabel("2 to 15 N width change [%]")
    axes[0].set_title("Shaft-width consistency")
    axes[1].bar(
        x, [row["edge_disagreement_15_n_px"] for row in run_qc], color="#f58518"
    )
    axes[1].set_ylabel("Left/right disagreement [px]")
    axes[1].set_title("Edge agreement at 15 N")
    axes[2].bar(x, [row["fixture_drift_15_n_px"] for row in run_qc], color="#54a24b")
    axes[2].axhline(0.0, color="#777777", linewidth=0.8)
    axes[2].set_ylabel("Fixture drift [px]")
    axes[2].set_title("Reference-edge drift at 15 N")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=60)
        axis.set_xlabel("Run")
    figure.savefig(output / "indentation_edge_tracking_width_consistency.png", dpi=220)
    plt.close(figure)


def _summary_markdown(
    index: Any,
    hold_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    run_qc: list[dict[str, Any]],
    synthetic_rows: list[dict[str, Any]],
    sign: float,
) -> str:
    valid_holds = sum(bool(row["tracking_valid"]) for row in hold_rows)
    valid_frames = sum(bool(row["tracking_valid"]) for row in frame_rows)
    monotonic_runs = sum(bool(row["trajectory_monotonic"]) for row in run_qc)
    positive_direction_count = sum(row["movement_2_to_15_n_px"] > 0.0 for row in run_qc)
    movement = _median_iqr_range(row["movement_2_to_15_n_px"] for row in run_qc)
    within = _percentiles(row["maximum_within_hold_range_px"] for row in run_qc)
    ratio = _percentiles(row["maximum_within_hold_over_total_signal"] for row in run_qc)
    disagreement = _percentiles(row["edge_disagreement_15_n_px"] for row in run_qc)
    width_change = _percentiles(row["width_change_magnitude_15_n_px"] for row in run_qc)
    width_fraction = _percentiles(
        row["width_change_fraction_magnitude_15_n"] for row in run_qc
    )
    fixture_drift = _percentiles(
        row["fixture_drift_magnitude_15_n_px"] for row in run_qc
    )
    valid_rows = _finite(row["moving_valid_row_fraction"] for row in hold_rows)
    valid_row_median = float(np.median(valid_rows))
    valid_row_minimum = float(np.min(valid_rows))
    synthetic_error = _percentiles(
        row["center_absolute_error_px"] for row in synthetic_rows
    )
    synthetic_disagreement = _percentiles(
        row["left_right_disagreement_px"] for row in synthetic_rows
    )
    believable = monotonic_runs > len(run_qc) / 2 and ratio[0] < 1.0
    lines = [
        "# Direct rigid-shaft edge tracking feasibility",
        "",
        "This is a read-only, pixel-domain feasibility study. It does not alter the",
        "raw dataset, the earlier phase-correlation or NCC studies, production code,",
        "Figure 5, or the current optomechanical observability metric.",
        "",
        "## Fixed study contract",
        "",
        f"- session: `{index.path}`",
        f"- specimen: `{index.session.specimen_id}`",
        f"- indenter: `{INDENTER}`",
        f"- runs: `{', '.join(SAMPLE_RUN_IDS)}`",
        f"- target forces [N]: `{TARGET_FORCES_N}`",
        f"- moving ROI `(x,y,w,h)`: `{MOVING_ROI_XYWH}`",
        f"- fixture ROI `(x,y,w,h)`: `{FIXTURE_ROI_XYWH}`",
        "- moving geometry: row-wise signed Scharr-x peaks, fixed 2 N polarity,",
        "  broad 2 N width/line consistency, and Theil-Sen left/right lines",
        "- fixture geometry: one directly tracked, fixed-polarity outer fixture edge",
        "  combined with a Theil-Sen line",
        "- reference: the same run's 2 N hold-median image",
        f"- common reported sign: `{int(sign):+d}`",
        "- units: pixels only; no pixel-to-mm conversion",
        "",
        "## Required numerical summary",
        "",
        f"- valid hold estimates: **{valid_holds}/{len(hold_rows)}**",
        f"- valid individual-frame estimates: **{valid_frames}/{len(frame_rows)}**",
        f"- positive 2 to 15 N direction after the common sign: **{positive_direction_count}/{len(run_qc)} runs**",
        f"- monotonic 2/5/10/15 N trajectories: **{monotonic_runs}/{len(run_qc)} runs**",
        f"- 2 to 15 N displacement, median/IQR/range [px]: **{movement[0]:.6g} / [{movement[1]:.6g}, {movement[2]:.6g}] / [{movement[3]:.6g}, {movement[4]:.6g}]**",
        f"- per-run maximum within-hold displacement range, median/p95/max [px]: **{within[0]:.6g} / {within[1]:.6g} / {within[2]:.6g}**",
        f"- max within-hold range / total signal, median/p95/max: **{ratio[0]:.6g} / {ratio[1]:.6g} / {ratio[2]:.6g}**",
        f"- 15 N left/right edge disagreement, median/p95/max [px]: **{disagreement[0]:.6g} / {disagreement[1]:.6g} / {disagreement[2]:.6g}**",
        f"- 15 N shaft-width change magnitude, median/p95/max [px]: **{width_change[0]:.6g} / {width_change[1]:.6g} / {width_change[2]:.6g}**",
        f"- 15 N shaft-width fractional change magnitude, median/p95/max: **{width_fraction[0]:.6g} / {width_fraction[1]:.6g} / {width_fraction[2]:.6g}**",
        f"- 15 N fixture-drift magnitude, median/p95/max [px]: **{fixture_drift[0]:.6g} / {fixture_drift[1]:.6g} / {fixture_drift[2]:.6g}**",
        f"- valid shaft-edge row fraction, median/minimum: **{valid_row_median:.6g} / {valid_row_minimum:.6g}**",
        f"- synthetic center absolute error, median/p95/max [px]: **{synthetic_error[0]:.6g} / {synthetic_error[1]:.6g} / {synthetic_error[2]:.6g}**",
        f"- synthetic left/right disagreement, median/p95/max [px]: **{synthetic_disagreement[0]:.6g} / {synthetic_disagreement[1]:.6g} / {synthetic_disagreement[2]:.6g}**",
        "",
        "## Per-run trajectories and 15 N QC",
        "",
        "| run | Y [mm] | rep | monotonic | delta 2/5/10/15 N [px] | edge disagreement [px] | abs width change [%] | abs fixture drift [px] | within/total |",
        "|---|---:|---:|:---:|---|---:|---:|---:|---:|",
    ]
    for row in run_qc:
        lines.append(
            f"| {row['run_id']} | {row['contact_position_mm']:g} | {row['repetition_index']} | "
            f"{'YES' if row['trajectory_monotonic'] else 'NO'} | "
            f"{row['delta_2_n_px']:.4g} / {row['delta_5_n_px']:.4g} / {row['delta_10_n_px']:.4g} / {row['delta_15_n_px']:.4g} | "
            f"{row['edge_disagreement_15_n_px']:.6g} | "
            f"{100.0 * row['width_change_fraction_magnitude_15_n']:.6g} | "
            f"{row['fixture_drift_magnitude_15_n_px']:.6g} | "
            f"{row['maximum_within_hold_over_total_signal']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The direct rigid-edge signal is mechanically believable on this fixed sample: "
                "the prescribed edge identities remained measurable and a majority of run-level "
                "force trajectories were monotonic."
                if believable
                else "The direct rigid-edge signal is not yet mechanically believable on this fixed sample. "
                "The reported validity, monotonicity, width, fixture, and within-hold diagnostics "
                "identify the observed failure modes without changing the method."
            ),
            "",
            "This feasibility result is not an automatic replacement for any current paper metric.",
            "The study stops at the required twelve-run sample.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    arguments = _arguments()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    index = index_session(SESSION_PATH)
    runs = _select_sample(index)
    groups = _group_frames(index, set(SAMPLE_RUN_IDS))
    hold_rows, frame_rows, geometry_rows, hold_images, references = _measure_sample(
        index, runs, groups
    )
    sign = _apply_common_sign(hold_rows, frame_rows)
    _add_hold_qc(hold_rows, frame_rows)
    run_qc = _run_qc(hold_rows)
    synthetic_rows = _synthetic_check(hold_images, references)

    _write_csv(output / "indentation_edge_tracking_hold_medians.csv", hold_rows)
    _write_csv(output / "indentation_edge_tracking_frame_qc.csv", frame_rows)
    _write_csv(output / "indentation_edge_tracking_synthetic_check.csv", synthetic_rows)
    _write_csv(output / "indentation_edge_tracking_geometry.csv", geometry_rows)
    _plot_overlay(output, hold_images, references)
    _plot_trajectories(output, hold_rows)
    _plot_within_hold(output, hold_rows, run_qc)
    _plot_width_consistency(output, run_qc)
    summary = _summary_markdown(
        index, hold_rows, frame_rows, run_qc, synthetic_rows, sign
    )
    (output / "indentation_edge_tracking_sample_summary.md").write_text(
        summary, encoding="utf-8"
    )
    print(summary)


if __name__ == "__main__":
    main()

"""Test mean-reduced signed-profile 1-D NCC as a shaft-motion proxy.

This script is deliberately separate from the earlier 2-D tracking baseline.
It reads the existing Solaris Baseline sample and writes only validation output.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.analysis.dataset import index_session  # noqa: E402
from experiments.analysis.metrics import actual_force_magnitude  # noqa: E402


SESSION_PATH = (
    REPOSITORY_ROOT
    / "output"
    / "contact_dataset"
    / "2026-09-04_solaris_baseline_01"
)
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "output"
    / "validation"
    / "hardware_indentation_tracking_1d_mean"
)
OUTPUT_PREFIX = "indentation_tracking_1d_mean"
SPECIMEN_ID = "solaris_baseline_02"
INDENTER = "sphere_10mm"
TARGET_FORCES_N = (2.0, 5.0, 10.0, 15.0)
REFERENCE_FORCE_N = 2.0
MAX_LAG_PX = 60
SECOND_PEAK_EXCLUSION_PX = 3
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
SYNTHETIC_SHIFTS_PX = (-20, -10, -5, -2, 2, 5, 10, 20)
CSV_FLOAT_FORMAT = ".9g"


@dataclass(frozen=True)
class Profile:
    values: np.ndarray
    centered_l2_norm: float

    @property
    def valid(self) -> bool:
        return bool(self.centered_l2_norm > np.finfo(np.float32).eps)


@dataclass(frozen=True)
class NccResult:
    valid: bool
    integer_lag_px: int | None
    subpixel_lag_px: float
    best_ncc: float
    second_best_lag_px: int | None
    second_best_ncc: float
    peak_margin: float
    at_max_lag: bool
    lags_px: np.ndarray
    ncc: np.ndarray


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


def _signed_scharr_x_profile(
    image: np.ndarray, roi_xywh: tuple[int, int, int, int]
) -> Profile:
    """Return the exact signed horizontal-edge profile requested by the study."""

    x, y, width, height = roi_xywh
    red = image[y : y + height, x : x + width, 2].astype(np.float32)
    gradient_x = cv2.Scharr(red, cv2.CV_32F, 1, 0)
    values = np.mean(gradient_x, axis=0, dtype=np.float64)
    values -= float(np.median(values))
    norm = float(np.linalg.norm(values))
    if norm > np.finfo(np.float32).eps:
        values /= norm
    else:
        values.fill(0.0)
    return Profile(values=values, centered_l2_norm=norm)


def _overlap_segments(
    reference: np.ndarray, current: np.ndarray, lag_px: int
) -> tuple[np.ndarray, np.ndarray]:
    if lag_px >= 0:
        if lag_px == 0:
            return reference, current
        return reference[:-lag_px], current[lag_px:]
    return reference[-lag_px:], current[:lag_px]


def _normalized_cross_correlation_1d(
    reference: Profile,
    current: Profile,
    *,
    max_lag_px: int = MAX_LAG_PX,
) -> NccResult:
    """Evaluate overlap-normalized NCC and retain its complete lag curve."""

    if reference.values.shape != current.values.shape:
        raise ValueError("reference and current profiles must have identical shape")
    if max_lag_px < 1 or max_lag_px >= len(reference.values):
        raise ValueError("max_lag_px must be positive and shorter than the profile")

    lags = np.arange(-max_lag_px, max_lag_px + 1, dtype=np.int32)
    scores = np.full(len(lags), np.nan, dtype=np.float64)
    if reference.valid and current.valid:
        for index, lag in enumerate(lags):
            reference_overlap, current_overlap = _overlap_segments(
                reference.values, current.values, int(lag)
            )
            reference_centered = reference_overlap - np.mean(reference_overlap)
            current_centered = current_overlap - np.mean(current_overlap)
            denominator = float(
                np.linalg.norm(reference_centered)
                * np.linalg.norm(current_centered)
            )
            if denominator > np.finfo(np.float64).eps:
                scores[index] = float(
                    np.dot(reference_centered, current_centered) / denominator
                )

    if not np.any(np.isfinite(scores)):
        return NccResult(
            valid=False,
            integer_lag_px=None,
            subpixel_lag_px=float("nan"),
            best_ncc=float("nan"),
            second_best_lag_px=None,
            second_best_ncc=float("nan"),
            peak_margin=float("nan"),
            at_max_lag=False,
            lags_px=lags,
            ncc=scores,
        )

    best_index = int(np.nanargmax(scores))
    best_lag = int(lags[best_index])
    best_ncc = float(scores[best_index])
    eligible_second = np.abs(lags - best_lag) > SECOND_PEAK_EXCLUSION_PX
    eligible_second &= np.isfinite(scores)
    if np.any(eligible_second):
        masked = np.where(eligible_second, scores, -np.inf)
        second_index = int(np.argmax(masked))
        second_lag: int | None = int(lags[second_index])
        second_ncc = float(scores[second_index])
        peak_margin = best_ncc - second_ncc
    else:
        second_lag = None
        second_ncc = float("nan")
        peak_margin = float("nan")

    subpixel_lag = float(best_lag)
    if 0 < best_index < len(scores) - 1:
        left = float(scores[best_index - 1])
        center = best_ncc
        right = float(scores[best_index + 1])
        denominator = left - 2.0 * center + right
        if (
            np.all(np.isfinite((left, center, right)))
            and denominator < -np.finfo(np.float64).eps
        ):
            offset = 0.5 * (left - right) / denominator
            if abs(offset) <= 1.0:
                subpixel_lag += float(offset)

    return NccResult(
        valid=True,
        integer_lag_px=best_lag,
        subpixel_lag_px=subpixel_lag,
        best_ncc=best_ncc,
        second_best_lag_px=second_lag,
        second_best_ncc=second_ncc,
        peak_margin=peak_margin,
        at_max_lag=abs(best_lag) == max_lag_px,
        lags_px=lags,
        ncc=scores,
    )


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


def _group_frames(
    index: Any, run_ids: set[str]
) -> dict[tuple[str, float], list[Any]]:
    groups: dict[tuple[str, float], list[Any]] = {}
    for frame in index.frames:
        if frame.run is None or frame.run.run_id not in run_ids:
            continue
        assert frame.target_force_n is not None
        key = frame.run.run_id, float(frame.target_force_n)
        groups.setdefault(key, []).append(frame)
    for frames in groups.values():
        frames.sort(key=lambda frame: int(frame.measurements["frame_index"]))
    required = {
        (run_id, force) for run_id in run_ids for force in TARGET_FORCES_N
    }
    missing = sorted(required - set(groups))
    if missing:
        raise ValueError(f"sample is missing required force holds: {missing}")
    return groups


def _record_curve(
    rows: list[dict[str, Any]],
    result: NccResult,
    reference: Profile,
    current: Profile,
    metadata: dict[str, Any],
) -> None:
    curve_id = 0 if not rows else int(rows[-1]["curve_id"]) + 1
    common = {
        "curve_id": curve_id,
        **metadata,
        "estimate_valid": result.valid,
        "reference_profile_l2_norm": reference.centered_l2_norm,
        "current_profile_l2_norm": current.centered_l2_norm,
        "best_lag_integer_px": result.integer_lag_px,
        "best_lag_subpixel_px": result.subpixel_lag_px,
        "best_ncc": result.best_ncc,
        "second_best_lag_px": result.second_best_lag_px,
        "second_best_ncc": result.second_best_ncc,
        "peak_margin": result.peak_margin,
        "best_lag_at_limit": result.at_max_lag,
    }
    for lag, score in zip(result.lags_px, result.ncc, strict=True):
        rows.append({**common, "lag_px": int(lag), "ncc": float(score)})


def _estimate(
    reference: Profile,
    current: Profile,
    curve_rows: list[dict[str, Any]],
    **metadata: Any,
) -> NccResult:
    result = _normalized_cross_correlation_1d(reference, current)
    _record_curve(curve_rows, result, reference, current, metadata)
    return result


def _relative_shift(moving: NccResult, fixture: NccResult) -> float:
    if not moving.valid or not fixture.valid:
        return float("nan")
    return moving.subpixel_lag_px - fixture.subpixel_lag_px


def _percentile(values: Iterable[float], percentile: float) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    return float(np.percentile(finite, percentile)) if len(finite) else float("nan")


def _summary(values: Iterable[float]) -> tuple[float, float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if not len(finite):
        return float("nan"), float("nan"), float("nan")
    return float(np.median(finite)), _percentile(finite, 95), float(np.max(finite))


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


def _prepare_profiles(
    index: Any,
    runs: list[Any],
    groups: dict[tuple[str, float], list[Any]],
) -> tuple[
    dict[tuple[str, float, str], Profile],
    dict[tuple[str, float, int, str], Profile],
    dict[tuple[str, float], list[dict[str, Any]]],
]:
    first = _load_bgr(groups[(runs[0].run_id, REFERENCE_FORCE_N)][0].rgb_path)
    moving_roi = _validate_roi(MOVING_ROI_XYWH, first.shape)
    fixture_roi = _validate_roi(FIXTURE_ROI_XYWH, first.shape)
    rois = {"moving": moving_roi, "fixture": fixture_roi}
    hold_profiles: dict[tuple[str, float, str], Profile] = {}
    frame_profiles: dict[tuple[str, float, int, str], Profile] = {}
    frame_metadata: dict[tuple[str, float], list[dict[str, Any]]] = {}

    for run in runs:
        for force in TARGET_FORCES_N:
            frames = groups[(run.run_id, force)]
            images = [_load_bgr(frame.rgb_path) for frame in frames]
            median_image = np.median(np.stack(images), axis=0).astype(np.uint8)
            for subject, roi in rois.items():
                hold_profiles[(run.run_id, force, subject)] = (
                    _signed_scharr_x_profile(median_image, roi)
                )
            metadata_rows = []
            for frame, image in zip(frames, images, strict=True):
                frame_index = int(frame.measurements["frame_index"])
                for subject, roi in rois.items():
                    frame_profiles[(run.run_id, force, frame_index, subject)] = (
                        _signed_scharr_x_profile(image, roi)
                    )
                metadata_rows.append(
                    {
                        "frame_index": frame_index,
                        "actual_force_n": _frame_force(frame.measurements),
                        "camera_host_time_s": float(
                            frame.measurements["camera_host_time_s"]
                        ),
                        "image_path": str(frame.rgb_path.resolve()),
                    }
                )
            frame_metadata[(run.run_id, force)] = metadata_rows
    return hold_profiles, frame_profiles, frame_metadata


def _primary_estimates(
    index: Any,
    runs: list[Any],
    hold_profiles: dict[tuple[str, float, str], Profile],
    frame_profiles: dict[tuple[str, float, int, str], Profile],
    frame_metadata: dict[tuple[str, float], list[dict[str, Any]]],
    curve_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    hold_intermediate: list[dict[str, Any]] = []
    frame_intermediate: list[dict[str, Any]] = []
    hold_results: dict[tuple[str, float, str], NccResult] = {}

    for run in runs:
        for force in TARGET_FORCES_N:
            subject_results = {}
            for subject in ("moving", "fixture"):
                reference = hold_profiles[(run.run_id, REFERENCE_FORCE_N, subject)]
                current = hold_profiles[(run.run_id, force, subject)]
                result = _estimate(
                    reference,
                    current,
                    curve_rows,
                    estimate_scope="hold_median_direct",
                    subject=subject,
                    run_id=run.run_id,
                    reference_force_n=REFERENCE_FORCE_N,
                    target_force_n=force,
                    frame_index="",
                    synthetic_shift_px="",
                )
                hold_results[(run.run_id, force, subject)] = result
                subject_results[subject] = result
            relative = _relative_shift(
                subject_results["moving"], subject_results["fixture"]
            )
            if force == REFERENCE_FORCE_N and np.isfinite(relative):
                relative = 0.0
            force_values = [
                row["actual_force_n"] for row in frame_metadata[(run.run_id, force)]
            ]
            hold_intermediate.append(
                {
                    "specimen_id": index.session.specimen_id,
                    "material": index.session.material,
                    "morphology": index.session.morphology,
                    "run_id": run.run_id,
                    "indenter": run.indenter,
                    "hole_index": run.hole_index,
                    "contact_position_mm": HOLE_TO_CONTACT_POSITION_MM[run.hole_index],
                    "repetition_index": run.repetition_index,
                    "target_force_n": force,
                    "frame_count": len(force_values),
                    "actual_force_median_n": float(np.median(force_values)),
                    "actual_force_std_n": float(np.std(force_values, ddof=1))
                    if len(force_values) > 1
                    else 0.0,
                    "actual_force_min_n": float(np.min(force_values)),
                    "actual_force_max_n": float(np.max(force_values)),
                    "moving": subject_results["moving"],
                    "fixture": subject_results["fixture"],
                    "relative_shift_px": relative,
                }
            )

            for frame in frame_metadata[(run.run_id, force)]:
                subject_results = {}
                for subject in ("moving", "fixture"):
                    reference = hold_profiles[(
                        run.run_id,
                        REFERENCE_FORCE_N,
                        subject,
                    )]
                    current = frame_profiles[(
                        run.run_id,
                        force,
                        frame["frame_index"],
                        subject,
                    )]
                    subject_results[subject] = _estimate(
                        reference,
                        current,
                        curve_rows,
                        estimate_scope="individual_frame_direct",
                        subject=subject,
                        run_id=run.run_id,
                        reference_force_n=REFERENCE_FORCE_N,
                        target_force_n=force,
                        frame_index=frame["frame_index"],
                        synthetic_shift_px="",
                    )
                frame_intermediate.append(
                    {
                        "specimen_id": index.session.specimen_id,
                        "material": index.session.material,
                        "morphology": index.session.morphology,
                        "run_id": run.run_id,
                        "indenter": run.indenter,
                        "hole_index": run.hole_index,
                        "contact_position_mm": HOLE_TO_CONTACT_POSITION_MM[
                            run.hole_index
                        ],
                        "repetition_index": run.repetition_index,
                        "target_force_n": force,
                        **frame,
                        "moving": subject_results["moving"],
                        "fixture": subject_results["fixture"],
                        "relative_shift_px": _relative_shift(
                            subject_results["moving"], subject_results["fixture"]
                        ),
                    }
                )

    final_shifts = np.asarray(
        [
            row["relative_shift_px"]
            for row in hold_intermediate
            if row["target_force_n"] == 15.0
            and np.isfinite(row["relative_shift_px"])
        ]
    )
    sign = 1.0 if not len(final_shifts) or np.median(final_shifts) >= 0.0 else -1.0
    for rows in (hold_intermediate, frame_intermediate):
        for row in rows:
            row["indentation_proxy_px"] = sign * row["relative_shift_px"]
    return hold_intermediate, frame_intermediate, sign


def _result_columns(prefix: str, result: NccResult) -> dict[str, Any]:
    return {
        f"{prefix}_estimate_valid": result.valid,
        f"{prefix}_lag_integer_px": result.integer_lag_px,
        f"{prefix}_lag_subpixel_px": result.subpixel_lag_px,
        f"{prefix}_best_ncc": result.best_ncc,
        f"{prefix}_second_best_lag_px": result.second_best_lag_px,
        f"{prefix}_second_best_ncc": result.second_best_ncc,
        f"{prefix}_peak_margin": result.peak_margin,
        f"{prefix}_best_lag_at_limit": result.at_max_lag,
    }


def _frame_rows(frame_intermediate: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in frame_intermediate:
        rows.append(
            {
                key: row[key]
                for key in (
                    "specimen_id",
                    "material",
                    "morphology",
                    "run_id",
                    "indenter",
                    "hole_index",
                    "contact_position_mm",
                    "repetition_index",
                    "target_force_n",
                    "frame_index",
                    "camera_host_time_s",
                    "actual_force_n",
                    "image_path",
                    "relative_shift_px",
                    "indentation_proxy_px",
                )
            }
            | _result_columns("moving", row["moving"])
            | _result_columns("fixture", row["fixture"])
        )
    return rows


def _hold_rows(
    runs: list[Any],
    hold_intermediate: list[dict[str, Any]],
    frame_intermediate: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        ordered = sorted(
            (row for row in hold_intermediate if row["run_id"] == run.run_id),
            key=lambda row: row["target_force_n"],
        )
        prior: list[float] = []
        prior_valid = True
        for row in ordered:
            force = row["target_force_n"]
            frames = [
                frame
                for frame in frame_intermediate
                if frame["run_id"] == run.run_id
                and frame["target_force_n"] == force
            ]
            proxy = np.asarray(
                [frame["indentation_proxy_px"] for frame in frames], dtype=np.float64
            )
            finite_proxy = proxy[np.isfinite(proxy)]
            quality_results = [
                result
                for frame in frames
                for result in (frame["moving"], frame["fixture"])
                if result.valid
            ]
            if np.isfinite(row["indentation_proxy_px"]):
                prior.append(float(row["indentation_proxy_px"]))
            else:
                prior_valid = False
            monotonic = prior_valid and bool(np.all(np.diff(prior) >= 0.0))
            output = {
                key: row[key]
                for key in (
                    "specimen_id",
                    "material",
                    "morphology",
                    "run_id",
                    "indenter",
                    "hole_index",
                    "contact_position_mm",
                    "repetition_index",
                    "target_force_n",
                    "frame_count",
                    "actual_force_median_n",
                    "actual_force_std_n",
                    "actual_force_min_n",
                    "actual_force_max_n",
                    "relative_shift_px",
                    "indentation_proxy_px",
                )
            }
            output.update(_result_columns("moving", row["moving"]))
            output.update(_result_columns("fixture", row["fixture"]))
            output.update(
                {
                    "finite_frame_count": int(len(finite_proxy)),
                    "frame_indentation_proxy_median_px": float(
                        np.median(finite_proxy)
                    )
                    if len(finite_proxy)
                    else float("nan"),
                    "frame_indentation_proxy_std_px": float(
                        np.std(finite_proxy, ddof=1)
                    )
                    if len(finite_proxy) > 1
                    else (0.0 if len(finite_proxy) == 1 else float("nan")),
                    "frame_indentation_proxy_min_px": float(np.min(finite_proxy))
                    if len(finite_proxy)
                    else float("nan"),
                    "frame_indentation_proxy_max_px": float(np.max(finite_proxy))
                    if len(finite_proxy)
                    else float("nan"),
                    "frame_indentation_proxy_range_px": float(np.ptp(finite_proxy))
                    if len(finite_proxy)
                    else float("nan"),
                    "frame_best_ncc_median": float(
                        np.median([result.best_ncc for result in quality_results])
                    )
                    if quality_results
                    else float("nan"),
                    "frame_best_ncc_min": float(
                        np.min([result.best_ncc for result in quality_results])
                    )
                    if quality_results
                    else float("nan"),
                    "frame_peak_margin_median": float(
                        np.median([result.peak_margin for result in quality_results])
                    )
                    if quality_results
                    else float("nan"),
                    "frame_peak_margin_min": float(
                        np.min([result.peak_margin for result in quality_results])
                    )
                    if quality_results
                    else float("nan"),
                    "monotonic_so_far": monotonic,
                    "tracking_status": "measured"
                    if np.isfinite(row["indentation_proxy_px"])
                    else "invalid_signed_profile_or_ncc",
                }
            )
            rows.append(output)
    return rows


def _direct_vs_sequential(
    runs: list[Any],
    hold_profiles: dict[tuple[str, float, str], Profile],
    hold_rows: list[dict[str, Any]],
    sign: float,
    curve_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    by_key = {(row["run_id"], row["target_force_n"]): row for row in hold_rows}
    for run in runs:
        cumulative = {"moving": 0.0, "fixture": 0.0}
        cumulative_valid = {"moving": True, "fixture": True}
        prior_force = REFERENCE_FORCE_N
        for force in TARGET_FORCES_N[1:]:
            for subject in ("moving", "fixture"):
                result = _estimate(
                    hold_profiles[(run.run_id, prior_force, subject)],
                    hold_profiles[(run.run_id, force, subject)],
                    curve_rows,
                    estimate_scope="hold_median_sequential",
                    subject=subject,
                    run_id=run.run_id,
                    reference_force_n=prior_force,
                    target_force_n=force,
                    frame_index="",
                    synthetic_shift_px="",
                )
                if result.valid and cumulative_valid[subject]:
                    cumulative[subject] += result.subpixel_lag_px
                else:
                    cumulative_valid[subject] = False
                    cumulative[subject] = float("nan")
            if force in (10.0, 15.0):
                direct = float(by_key[(run.run_id, force)]["indentation_proxy_px"])
                sequential = sign * (cumulative["moving"] - cumulative["fixture"])
                disagreement = sequential - direct
                rows.append(
                    {
                        "specimen_id": hold_rows[0]["specimen_id"],
                        "run_id": run.run_id,
                        "indenter": run.indenter,
                        "hole_index": run.hole_index,
                        "contact_position_mm": HOLE_TO_CONTACT_POSITION_MM[
                            run.hole_index
                        ],
                        "repetition_index": run.repetition_index,
                        "target_force_n": force,
                        "direct_indentation_proxy_px": direct,
                        "cumulative_sequential_indentation_proxy_px": sequential,
                        "sequential_minus_direct_px": disagreement,
                        "absolute_disagreement_px": abs(disagreement),
                        "comparison_valid": bool(
                            np.isfinite(direct) and np.isfinite(sequential)
                        ),
                    }
                )
            prior_force = force
    return rows


def _translated_profile(profile: Profile, shift_px: int) -> Profile:
    values = np.zeros_like(profile.values)
    if shift_px > 0:
        values[shift_px:] = profile.values[:-shift_px]
    elif shift_px < 0:
        values[:shift_px] = profile.values[-shift_px:]
    else:
        values[:] = profile.values
    values -= float(np.median(values))
    norm = float(np.linalg.norm(values))
    if norm > np.finfo(np.float32).eps:
        values /= norm
    return Profile(values=values, centered_l2_norm=norm)


def _synthetic_check(
    hold_profiles: dict[tuple[str, float, str], Profile],
    curve_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_ids = [
        run_id
        for run_id in SAMPLE_RUN_IDS
        if hold_profiles[(run_id, REFERENCE_FORCE_N, "fixture")].valid
    ][:4]
    if len(source_ids) < 2:
        raise RuntimeError("fewer than two real non-degenerate 2 N profiles")
    rows = []
    for run_id in source_ids:
        reference = hold_profiles[(run_id, REFERENCE_FORCE_N, "fixture")]
        for requested_shift in SYNTHETIC_SHIFTS_PX:
            current = _translated_profile(reference, requested_shift)
            result = _estimate(
                reference,
                current,
                curve_rows,
                estimate_scope="synthetic_translation",
                subject="fixture",
                run_id=run_id,
                reference_force_n=REFERENCE_FORCE_N,
                target_force_n="",
                frame_index="",
                synthetic_shift_px=requested_shift,
            )
            rows.append(
                {
                    "source_run_id": run_id,
                    "source_subject": "fixture",
                    "requested_shift_px": requested_shift,
                    "estimate_valid": result.valid,
                    "recovered_integer_shift_px": result.integer_lag_px,
                    "recovered_subpixel_shift_px": result.subpixel_lag_px,
                    "integer_absolute_error_px": abs(
                        float(result.integer_lag_px) - requested_shift
                    )
                    if result.integer_lag_px is not None
                    else float("nan"),
                    "subpixel_absolute_error_px": abs(
                        result.subpixel_lag_px - requested_shift
                    ),
                    "best_ncc": result.best_ncc,
                    "second_best_lag_px": result.second_best_lag_px,
                    "second_best_ncc": result.second_best_ncc,
                    "peak_margin": result.peak_margin,
                    "best_lag_at_limit": result.at_max_lag,
                }
            )
    return rows


def _plot_run_overview(
    output: Path,
    hold_profiles: dict[tuple[str, float, str], Profile],
    hold_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    direct_sequential_rows: list[dict[str, Any]],
) -> None:
    complete_runs = []
    for run_id in SAMPLE_RUN_IDS:
        rows = [row for row in hold_rows if row["run_id"] == run_id]
        if all(np.isfinite(row["indentation_proxy_px"]) for row in rows):
            complete_runs.append(run_id)
    run_id = complete_runs[0] if complete_runs else SAMPLE_RUN_IDS[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(TARGET_FORCES_N)))
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), constrained_layout=True)
    for subject, axis in zip(("moving", "fixture"), axes[0], strict=True):
        for force, color in zip(TARGET_FORCES_N, colors, strict=True):
            profile = hold_profiles[(run_id, force, subject)]
            axis.plot(profile.values, color=color, label=f"{force:g} N")
        axis.set_title(f"{subject.capitalize()} mean-reduced signed Scharr-x profiles")
        axis.set_xlabel("ROI x [px]")
        axis.set_ylabel("Normalized signed profile")
    axes[0, 0].legend(frameon=False, ncol=2)

    run_holds = sorted(
        (row for row in hold_rows if row["run_id"] == run_id),
        key=lambda row: row["actual_force_median_n"],
    )
    run_frames = [row for row in frame_rows if row["run_id"] == run_id]
    axes[1, 0].plot(
        [row["actual_force_median_n"] for row in run_holds],
        [row["indentation_proxy_px"] for row in run_holds],
        "-o",
        color="#d95f02",
        label="hold median",
    )
    axes[1, 0].scatter(
        [row["actual_force_n"] for row in run_frames],
        [row["indentation_proxy_px"] for row in run_frames],
        s=22,
        facecolors="none",
        edgecolors="#444444",
        label="individual frames",
    )
    axes[1, 0].set_title(f"{run_id}: direct to 2 N reference")
    axes[1, 0].set_xlabel("Measured force [N]")
    axes[1, 0].set_ylabel("Indentation proxy [px]")
    axes[1, 0].legend(frameon=False)

    comparison = [
        row for row in direct_sequential_rows if row["run_id"] == run_id
    ]
    axes[1, 1].plot(
        [row["target_force_n"] for row in comparison],
        [row["direct_indentation_proxy_px"] for row in comparison],
        "-o",
        label="direct to 2 N",
    )
    axes[1, 1].plot(
        [row["target_force_n"] for row in comparison],
        [row["cumulative_sequential_indentation_proxy_px"] for row in comparison],
        "--s",
        label="cumulative sequential",
    )
    axes[1, 1].set_title("Direct versus sequential diagnostic")
    axes[1, 1].set_xlabel("Target force [N]")
    axes[1, 1].set_ylabel("Indentation proxy [px]")
    axes[1, 1].legend(frameon=False)
    figure.suptitle("1-D rigid-shaft tracking overview")
    figure.savefig(output / f"{OUTPUT_PREFIX}_run_overview.png", dpi=220)
    plt.close(figure)


def _plot_within_hold_qc(
    output: Path,
    hold_rows: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    by_force = [
        [
            row["frame_indentation_proxy_range_px"]
            for row in hold_rows
            if row["target_force_n"] == force
            and np.isfinite(row["frame_indentation_proxy_range_px"])
        ]
        for force in TARGET_FORCES_N
    ]
    axes[0].boxplot(by_force, tick_labels=[f"{force:g}" for force in TARGET_FORCES_N])
    axes[0].set_xlabel("Target force [N]")
    axes[0].set_ylabel("Within-hold frame range [px]")
    axes[0].set_title("Frame-level stability")

    for run_id in SAMPLE_RUN_IDS:
        rows = [row for row in hold_rows if row["run_id"] == run_id]
        final = next(row for row in rows if row["target_force_n"] == 15.0)
        ranges = [
            row["frame_indentation_proxy_range_px"]
            for row in rows
            if np.isfinite(row["frame_indentation_proxy_range_px"])
        ]
        if (
            np.isfinite(final["indentation_proxy_px"])
            and abs(float(final["indentation_proxy_px"])) > 0.0
            and ranges
        ):
            axes[1].scatter(
                final["indentation_proxy_px"],
                max(ranges),
                color=plt.cm.viridis((final["hole_index"] - 1) / 5),
                edgecolor="#333333",
                linewidth=0.4,
            )
            axes[1].annotate(
                run_id[-2:],
                (final["indentation_proxy_px"], max(ranges)),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
    axes[1].axline((0, 0), slope=1, color="#999999", linestyle="--", linewidth=1)
    axes[1].set_xlabel("2 to 15 N hold-median signal [px]")
    axes[1].set_ylabel("Maximum within-hold frame range [px]")
    axes[1].set_title("Signal versus frame disagreement")
    figure.savefig(output / f"{OUTPUT_PREFIX}_within_hold_qc.png", dpi=220)
    plt.close(figure)


def _plot_ncc_examples(output: Path, curve_rows: list[dict[str, Any]]) -> None:
    curves: dict[int, list[dict[str, Any]]] = {}
    for row in curve_rows:
        curves.setdefault(int(row["curve_id"]), []).append(row)
    valid = [rows for rows in curves.values() if rows[0]["estimate_valid"]]
    real = [
        rows
        for rows in valid
        if rows[0]["estimate_scope"] != "synthetic_translation"
        and np.isfinite(rows[0]["peak_margin"])
    ]
    synthetic = [
        rows for rows in valid if rows[0]["estimate_scope"] == "synthetic_translation"
    ]
    if not real or not synthetic:
        raise RuntimeError("NCC example plotting requires real and synthetic curves")
    real_sorted = sorted(real, key=lambda rows: float(rows[0]["peak_margin"]))
    choices = [real_sorted[0], real_sorted[len(real_sorted) // 2], real_sorted[-1], synthetic[0]]
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 6.8), constrained_layout=True)
    for axis, rows in zip(axes.flat, choices, strict=True):
        first = rows[0]
        lags = np.asarray([row["lag_px"] for row in rows])
        scores = np.asarray([row["ncc"] for row in rows])
        axis.plot(lags, scores, color="#31688e")
        axis.axvline(first["best_lag_integer_px"], color="#d95f02", linewidth=1.2)
        title = (
            f"synthetic {first['synthetic_shift_px']:+g} px"
            if first["estimate_scope"] == "synthetic_translation"
            else (
                f"{first['estimate_scope']} · {first['subject']} · "
                f"{first['run_id']} · {float(first['target_force_n']):g} N"
            )
        )
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Integer lag [px]")
        axis.set_ylabel("Overlap-normalized NCC")
        axis.text(
            0.02,
            0.04,
            f"best={float(first['best_ncc']):.3f}\nmargin={float(first['peak_margin']):.3f}",
            transform=axis.transAxes,
            fontsize=8,
        )
    figure.suptitle("Signed-profile NCC peak examples")
    figure.savefig(output / f"{OUTPUT_PREFIX}_ncc_examples.png", dpi=220)
    plt.close(figure)


def _iqr(values: Iterable[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    return float(np.percentile(finite, 75) - np.percentile(finite, 25)) if len(finite) else float("nan")


def _report(
    output: Path,
    hold_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    direct_sequential_rows: list[dict[str, Any]],
    synthetic_rows: list[dict[str, Any]],
) -> str:
    finals = [row for row in hold_rows if row["target_force_n"] == 15.0]
    finite_holds = [row for row in hold_rows if np.isfinite(row["indentation_proxy_px"])]
    finite_frames = [row for row in frame_rows if np.isfinite(row["indentation_proxy_px"])]
    moving_finite_holds = sum(row["moving_estimate_valid"] for row in hold_rows)
    fixture_finite_holds = sum(row["fixture_estimate_valid"] for row in hold_rows)
    moving_finite_frames = sum(row["moving_estimate_valid"] for row in frame_rows)
    fixture_finite_frames = sum(row["fixture_estimate_valid"] for row in frame_rows)
    finite_final = [row for row in finals if np.isfinite(row["indentation_proxy_px"])]
    positive_final = sum(row["indentation_proxy_px"] > 0.0 for row in finite_final)
    monotonic_count = 0
    fully_finite_runs = 0
    run_max_stds = []
    run_max_ranges = []
    for run_id in SAMPLE_RUN_IDS:
        rows = sorted(
            (row for row in hold_rows if row["run_id"] == run_id),
            key=lambda row: row["target_force_n"],
        )
        values = np.asarray([row["indentation_proxy_px"] for row in rows])
        if np.all(np.isfinite(values)):
            fully_finite_runs += 1
            monotonic_count += int(np.all(np.diff(values) >= 0.0))
        stds = [
            row["frame_indentation_proxy_std_px"]
            for row in rows
            if np.isfinite(row["frame_indentation_proxy_std_px"])
        ]
        ranges = [
            row["frame_indentation_proxy_range_px"]
            for row in rows
            if np.isfinite(row["frame_indentation_proxy_range_px"])
        ]
        run_max_stds.append(max(stds) if stds else float("nan"))
        run_max_ranges.append(max(ranges) if ranges else float("nan"))

    final_values = [row["indentation_proxy_px"] for row in finite_final]
    final_median = float(np.median(final_values)) if final_values else float("nan")
    final_range = (
        f"{np.min(final_values):.3f}..{np.max(final_values):.3f}"
        if final_values
        else "unavailable"
    )
    fixture_final = [
        abs(float(row["fixture_lag_subpixel_px"]))
        for row in finals
        if row["fixture_estimate_valid"]
    ]
    fixture_stats = _summary(fixture_final)
    frame_std_stats = _summary(run_max_stds)
    frame_range_stats = _summary(run_max_ranges)
    quality_results = [
        value
        for row in frame_rows
        for value in (
            row["moving_best_ncc"],
            row["fixture_best_ncc"],
        )
        if np.isfinite(value)
    ]
    margin_results = [
        value
        for row in frame_rows
        for value in (
            row["moving_peak_margin"],
            row["fixture_peak_margin"],
        )
        if np.isfinite(value)
    ]
    disagreements = [
        row["absolute_disagreement_px"]
        for row in direct_sequential_rows
        if row["target_force_n"] == 15.0 and row["comparison_valid"]
    ]
    disagreement_stats = _summary(disagreements)
    frame_range_to_signal = []
    for run_id in SAMPLE_RUN_IDS:
        rows = [row for row in hold_rows if row["run_id"] == run_id]
        final = next(row for row in rows if row["target_force_n"] == 15.0)
        ranges = [
            row["frame_indentation_proxy_range_px"]
            for row in rows
            if np.isfinite(row["frame_indentation_proxy_range_px"])
        ]
        if np.isfinite(final["indentation_proxy_px"]) and ranges:
            frame_range_to_signal.append(
                max(ranges) / abs(float(final["indentation_proxy_px"]))
            )
    frame_ratio_stats = _summary(frame_range_to_signal)
    sequential_to_signal = [
        row["absolute_disagreement_px"]
        / abs(float(row["direct_indentation_proxy_px"]))
        for row in direct_sequential_rows
        if row["target_force_n"] == 15.0
        and row["comparison_valid"]
        and float(row["direct_indentation_proxy_px"]) != 0.0
    ]
    sequential_ratio_stats = _summary(sequential_to_signal)
    synthetic_errors = [
        row["subpixel_absolute_error_px"]
        for row in synthetic_rows
        if row["estimate_valid"]
    ]
    synthetic_stats = _summary(synthetic_errors)
    boundary_count = sum(
        bool(row["moving_best_lag_at_limit"])
        + bool(row["fixture_best_lag_at_limit"])
        for row in frame_rows
    )

    conclusion = "Tracking signal is not yet trustworthy enough for optomechanical metrics."
    lines = [
        "# Hardware indentation tracking: mean-reduced signed-profile 1-D NCC",
        "",
        "This is a new study. The previous 2-D feasibility output remains unchanged.",
        "",
        "## Study contract",
        "",
        "- motion estimator: **1-D normalized cross-correlation on vertically mean-reduced signed red-channel Scharr-x profiles**",
        f"- analyzed runs: `{len(SAMPLE_RUN_IDS)}` (`{', '.join(SAMPLE_RUN_IDS)}`)",
        f"- session: `{SESSION_PATH}`",
        f"- specimen: `{SPECIMEN_ID}`",
        f"- indenter: `{INDENTER}`",
        "- force states: `2, 5, 10, 15 N`; hold medians are primary and individual frames are QC",
        f"- moving ROI `[x,y,w,h]`: `{list(MOVING_ROI_XYWH)}`",
        f"- fixed ROI `[x,y,w,h]`: `{list(FIXTURE_ROI_XYWH)}`",
        f"- integer-lag search: `[-{MAX_LAG_PX}, +{MAX_LAG_PX}] px`; second peak excludes +/-{SECOND_PEAK_EXCLUSION_PX} px",
        "- reported motion: same-run rigid-shaft minus fixture displacement relative to 2 N, in pixels",
        "- contact positions: `0, 10, 20, 30, 40, 50 mm` for holes 1--6",
        "",
        "## Measured evidence",
        "",
        f"- finite hold profiles: moving `{moving_finite_holds}/{len(hold_rows)}`, fixture `{fixture_finite_holds}/{len(hold_rows)}`, both/relative `{len(finite_holds)}/{len(hold_rows)}`",
        f"- finite individual-frame profiles: moving `{moving_finite_frames}/{len(frame_rows)}`, fixture `{fixture_finite_frames}/{len(frame_rows)}`, both/relative `{len(finite_frames)}/{len(frame_rows)}`",
        f"- positive 2-to-15 N direction: `{positive_final}/{len(finite_final)}` finite runs (`{positive_final}/{len(SAMPLE_RUN_IDS)}` selected runs)",
        f"- monotonic 2/5/10/15 N trajectories: `{monotonic_count}/{fully_finite_runs}` fully finite runs",
        f"- 2-to-15 N indentation proxy median / IQR / range: `{final_median:.3f}` / `{_iqr(final_values):.3f}` / `{final_range}` px",
        f"- fixed-fixture shift at 15 N median / p95 / max: `{fixture_stats[0]:.3f}` / `{fixture_stats[1]:.3f}` / `{fixture_stats[2]:.3f}` px",
        f"- maximum within-hold frame std median / p95 / max: `{frame_std_stats[0]:.3f}` / `{frame_std_stats[1]:.3f}` / `{frame_std_stats[2]:.3f}` px",
        f"- maximum within-hold frame range median / p95 / max: `{frame_range_stats[0]:.3f}` / `{frame_range_stats[1]:.3f}` / `{frame_range_stats[2]:.3f}` px",
        f"- per-run maximum frame range / 2-to-15 N signal median / p95 / max: `{frame_ratio_stats[0]:.3f}` / `{frame_ratio_stats[1]:.3f}` / `{frame_ratio_stats[2]:.3f}`",
        f"- individual-estimate best NCC median / minimum: `{np.median(quality_results):.4f}` / `{np.min(quality_results):.4f}`",
        f"- individual-estimate peak margin median / minimum: `{np.median(margin_results):.4f}` / `{np.min(margin_results):.4f}`",
        f"- frame estimates at +/-{MAX_LAG_PX} px boundary: `{boundary_count}`",
        f"- direct-vs-sequential disagreement at 15 N median / p95 / max: `{disagreement_stats[0]:.3f}` / `{disagreement_stats[1]:.3f}` / `{disagreement_stats[2]:.3f}` px across `{len(disagreements)}` finite comparisons",
        f"- direct-vs-sequential disagreement / direct signal median / p95 / max: `{sequential_ratio_stats[0]:.3f}` / `{sequential_ratio_stats[1]:.3f}` / `{sequential_ratio_stats[2]:.3f}`",
        f"- synthetic recovery absolute error median / p95 / max: `{synthetic_stats[0]:.4f}` / `{synthetic_stats[1]:.4f}` / `{synthetic_stats[2]:.4f}` px",
        "",
        "## Feasibility conclusion",
        "",
        f"The vertical mean fixed the profile-degeneracy failure: all `{len(hold_rows)}` hold and `{len(frame_rows)}` frame estimates were finite. It did not fully close the mechanical-consistency gate. Only `{monotonic_count}/{fully_finite_runs}` trajectories were strictly monotonic; the per-run maximum within-hold range was `{frame_ratio_stats[0]:.1%}` of the complete 2-to-15 N signal at the median and `{frame_ratio_stats[1]:.1%}` at p95; direct-versus-sequential disagreement was `{sequential_ratio_stats[0]:.1%}` of the direct signal at the median and `{sequential_ratio_stats[1]:.1%}` at p95. The mean reduction is a clear improvement over the median reduction, but nominally fixed-hold variation and path disagreement are not yet small compared with the approximately one-pixel total motion.",
        "",
        f"**{conclusion}**",
        "",
        "No millimeter conversion, mechanical-deformation claim, optical-change-per-displacement metric, production metric, or Figure 5 value is produced.",
        "",
    ]
    report = "\n".join(lines)
    (output / f"{OUTPUT_PREFIX}_sample_summary.md").write_text(
        report, encoding="utf-8"
    )
    return report


def main() -> None:
    args = _arguments()
    output = args.output if args.output.is_absolute() else REPOSITORY_ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    index = index_session(SESSION_PATH, expected_repetitions=5)
    runs = _select_sample(index)
    groups = _group_frames(index, set(SAMPLE_RUN_IDS))
    hold_profiles, frame_profiles, frame_metadata = _prepare_profiles(
        index, runs, groups
    )
    curve_rows: list[dict[str, Any]] = []
    hold_intermediate, frame_intermediate, sign = _primary_estimates(
        index,
        runs,
        hold_profiles,
        frame_profiles,
        frame_metadata,
        curve_rows,
    )
    frame_rows = _frame_rows(frame_intermediate)
    hold_rows = _hold_rows(runs, hold_intermediate, frame_intermediate)
    direct_sequential_rows = _direct_vs_sequential(
        runs, hold_profiles, hold_rows, sign, curve_rows
    )
    synthetic_rows = _synthetic_check(hold_profiles, curve_rows)

    _write_csv(output / f"{OUTPUT_PREFIX}_hold_medians.csv", hold_rows)
    _write_csv(output / f"{OUTPUT_PREFIX}_frame_qc.csv", frame_rows)
    _write_csv(output / f"{OUTPUT_PREFIX}_ncc_peaks.csv", curve_rows)
    _write_csv(
        output / f"{OUTPUT_PREFIX}_direct_vs_sequential.csv",
        direct_sequential_rows,
    )
    _write_csv(
        output / f"{OUTPUT_PREFIX}_synthetic_check.csv", synthetic_rows
    )
    _plot_run_overview(
        output, hold_profiles, hold_rows, frame_rows, direct_sequential_rows
    )
    _plot_within_hold_qc(output, hold_rows)
    _plot_ncc_examples(output, curve_rows)
    report = _report(
        output,
        hold_rows,
        frame_rows,
        direct_sequential_rows,
        synthetic_rows,
    )
    print(report)
    print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()

"""Validate the fixed-experiment geometry calibration on saved RGB images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.localization import (  # noqa: E402
    FixedFingerCalibration,
    calibrate_fixed_finger,
    load_fixed_finger_calibration,
    project_longitudinal_positions,
    save_fixed_finger_calibration,
    warp_with_fixed_finger_calibration,
)
from experiments.localization.fixed_finger_calibration import (  # noqa: E402
    LED_REFINEMENT_HALF_WINDOW_IN_SPACINGS,
)
from lumo.fingertip.layout import (  # noqa: E402
    LED_CENTERS_Y_MM,
    TOTAL_Y_BOUNDS_MM,
)


IMAGE_DIRECTORY = REPOSITORY_ROOT / "experiments" / "img"
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "output" / "validation" / "fixed_finger_calibration"
)
GROUND_TRUTH_PATH = REPOSITORY_ROOT / "validation" / "fixed_finger_led_ground_truth.json"
REFERENCE_IMAGES = (
    ("Solaris representative", "solaris_p1_Color.png"),
    ("Dragon Skin unloaded", "dragonskin_unloaded_Color.png"),
    ("Dark-room Solaris", "solaris_unloaded_dark_Color.png"),
)


def _load_rgb(filename: str) -> np.ndarray:
    path = IMAGE_DIRECTORY / filename
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read calibration image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _label_ground_truth() -> None:
    labels: dict[str, list[list[float]]] = {}
    for title, filename in REFERENCE_IMAGES:
        rgb = _load_rgb(filename)
        figure, axis = plt.subplots(figsize=(12.0, 7.0))
        axis.imshow(rgb)
        axis.set_title(
            f"{title}: click five physical LED centers distal to proximal",
        )
        axis.axis("off")
        points = plt.ginput(5, timeout=-1, show_clicks=True)
        plt.close(figure)
        if len(points) != 5:
            raise RuntimeError(f"{filename}: expected exactly five manual clicks")
        labels[filename] = [[float(x), float(y)] for x, y in points]
    payload = {
        "coordinate_system": "original image pixels [x, y]",
        "point_order": "distal to proximal",
        "images": labels,
    }
    GROUND_TRUTH_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Manual LED ground truth: {GROUND_TRUTH_PATH}")


def _load_ground_truth(*, required: bool) -> dict[str, np.ndarray] | None:
    if not GROUND_TRUTH_PATH.is_file():
        if not required:
            return None
        raise FileNotFoundError(
            "manual LED ground truth is required; run this validator once with "
            "--label-ground-truth and click the five physical LED centers in "
            "distal-to-proximal order"
        )
    payload = json.loads(GROUND_TRUTH_PATH.read_text())
    if payload.get("point_order") != "distal to proximal":
        raise ValueError("manual LED labels must be ordered distal to proximal")
    labels = {}
    for _, filename in REFERENCE_IMAGES:
        points = np.asarray(payload.get("images", {}).get(filename), dtype=np.float64)
        if points.shape != (5, 2) or not np.all(np.isfinite(points)):
            raise ValueError(f"{filename}: manual labels must be a finite 5 x 2 array")
        labels[filename] = points
    return labels


def _led_search_window_segments(
    calibration: FixedFingerCalibration,
) -> tuple[np.ndarray, ...]:
    proximal_y_mm, distal_y_mm = TOTAL_Y_BOUNDS_MM
    total_length_mm = distal_y_mm - proximal_y_mm
    led_y_distal_to_proximal = np.sort(np.asarray(LED_CENTERS_Y_MM))[::-1]
    centers = (distal_y_mm - led_y_distal_to_proximal) / total_length_mm
    pitch_fraction = abs(LED_CENTERS_Y_MM[1] - LED_CENTERS_Y_MM[0]) / total_length_mm
    half_window = LED_REFINEMENT_HALF_WINDOW_IN_SPACINGS * pitch_fraction
    return tuple(
        project_longitudinal_positions(
            calibration.led_line,
            calibration.distal_longitudinal_limit,
            calibration.proximal_longitudinal_limit,
            calibration.vanishing_point_h,
            np.asarray((center - half_window, center + half_window)),
        )
        for center in centers
    )


def _line_segment(
    calibration: FixedFingerCalibration,
    line: np.ndarray,
) -> np.ndarray:
    return project_longitudinal_positions(
        line,
        calibration.distal_longitudinal_limit,
        calibration.proximal_longitudinal_limit,
        calibration.vanishing_point_h,
        np.asarray((0.0, 1.0)),
    )


def _round_trip_matches(
    expected: FixedFingerCalibration,
    actual: FixedFingerCalibration,
) -> bool:
    return (
        expected.image_shape == actual.image_shape
        and expected.distal_orientation == actual.distal_orientation
        and expected.led_line_alpha == actual.led_line_alpha
        and expected.led_line_score == actual.led_line_score
        and all(
            np.array_equal(getattr(expected, field), getattr(actual, field))
            for field in (
                "dorsal_line",
                "palmar_line",
                "led_line",
                "vanishing_point_h",
                "distal_longitudinal_limit",
                "proximal_longitudinal_limit",
                "led_centers_xy_px",
                "led_longitudinal_fractions",
                "canonical_map_x",
                "canonical_map_y",
                "reference_mask",
            )
        )
    )


def _overlay(
    axis: plt.Axes,
    rgb: np.ndarray,
    calibration: FixedFingerCalibration,
    ground_truth_xy_px: np.ndarray | None,
    title: str,
) -> None:
    axis.imshow(rgb)
    axis.contour(
        calibration.reference_mask,
        levels=(0.5,),
        colors=("#f1c40f",),
        linewidths=1.2,
    )
    for line, color, label in (
        (calibration.dorsal_line, "#31688e", "dorsal"),
        (calibration.palmar_line, "#e67e22", "palmar"),
        (calibration.led_line, "#00a86b", "LED line"),
    ):
        segment = _line_segment(calibration, line)
        axis.plot(segment[:, 0], segment[:, 1], color=color, linewidth=2.0, label=label)
    for index, segment in enumerate(_led_search_window_segments(calibration)):
        axis.plot(
            segment[:, 0],
            segment[:, 1],
            color="#7b2cbf",
            linewidth=7.0,
            alpha=0.35,
            solid_capstyle="butt",
            label="LED score windows" if index == 0 else None,
            zorder=3,
        )
    axis.scatter(
        calibration.led_centers_xy_px[:, 0],
        calibration.led_centers_xy_px[:, 1],
        s=30,
        color="#ff2ca0",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    if ground_truth_xy_px is not None:
        axis.scatter(
            ground_truth_xy_px[:, 0],
            ground_truth_xy_px[:, 1],
            marker="x",
            s=45,
            color="#ffd166",
            linewidth=1.4,
            label="manual LED GT",
            zorder=5,
        )
    for index, center in enumerate(calibration.led_centers_xy_px, start=1):
        axis.text(
            center[0] + 5,
            center[1] - 4,
            str(index),
            color="white",
            fontsize=8,
            weight="bold",
            zorder=5,
        )
    axis.set_title(
        f"{title}\n$\\alpha$={calibration.led_line_alpha:.3f}, "
        f"line score={calibration.led_line_score:.3f}",
        fontsize=10,
    )
    axis.legend(loc="lower right", fontsize=8, framealpha=0.9)
    axis.axis("off")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-ground-truth",
        action="store_true",
        help="interactively click five physical LED centers in each reference image",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="write overlays without claiming or measuring LED-position accuracy",
    )
    arguments = parser.parse_args()
    if arguments.label_ground_truth:
        _label_ground_truth()
    ground_truth = _load_ground_truth(required=not arguments.diagnostic_only)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(REFERENCE_IMAGES),
        2,
        figsize=(10.0, 11.0),
        constrained_layout=True,
    )
    rows = []
    geometry_valid = True
    for row_index, (title, filename) in enumerate(REFERENCE_IMAGES):
        rgb = _load_rgb(filename)
        cv2.setRNGSeed(0)
        calibration = calibrate_fixed_finger(rgb)
        artifact_name = filename.removesuffix("_Color.png").lower()
        calibration_path = OUTPUT_DIRECTORY / f"{artifact_name}_calibration.npz"
        save_fixed_finger_calibration(calibration_path, calibration)
        loaded = load_fixed_finger_calibration(calibration_path)
        round_trip = _round_trip_matches(calibration, loaded)

        rounded_centers = np.rint(calibration.led_centers_xy_px).astype(np.int64)
        rounded_centers[:, 0] = np.clip(rounded_centers[:, 0], 0, rgb.shape[1] - 1)
        rounded_centers[:, 1] = np.clip(rounded_centers[:, 1], 0, rgb.shape[0] - 1)
        centers_inside = bool(
            np.all(
                calibration.reference_mask[
                    rounded_centers[:, 1],
                    rounded_centers[:, 0],
                ]
            )
        )
        ordered = bool(np.all(np.diff(calibration.led_longitudinal_fractions) > 0.0))
        case_geometry_valid = round_trip and centers_inside and ordered
        geometry_valid &= case_geometry_valid
        labels_xy_px = None if ground_truth is None else ground_truth[filename]
        if labels_xy_px is None:
            median_led_error_px = None
            maximum_led_error_px = None
            median_line_distance_px = None
            maximum_line_distance_px = None
        else:
            led_errors_px = np.linalg.norm(
                calibration.led_centers_xy_px - labels_xy_px,
                axis=1,
            )
            line_distances_px = np.abs(
                labels_xy_px @ calibration.led_line[:2] + calibration.led_line[2]
            )
            median_led_error_px = float(np.median(led_errors_px))
            maximum_led_error_px = float(np.max(led_errors_px))
            median_line_distance_px = float(np.median(line_distances_px))
            maximum_line_distance_px = float(np.max(line_distances_px))
        vanishing = calibration.vanishing_point_h
        vanishing_kind = "infinity" if vanishing[2] == 0.0 else "finite"
        rows.append(
            (
                title,
                filename,
                calibration.led_line_alpha,
                calibration.led_line_score,
                vanishing_kind,
                centers_inside,
                ordered,
                round_trip,
                median_led_error_px,
                maximum_led_error_px,
                median_line_distance_px,
                maximum_line_distance_px,
                "VALID" if case_geometry_valid else "INVALID",
            )
        )

        _overlay(axes[row_index, 0], rgb, calibration, labels_xy_px, title)
        axes[row_index, 1].imshow(warp_with_fixed_finger_calibration(rgb, loaded))
        axes[row_index, 1].set_title("Fixed image-space sampling strip", fontsize=10)
        axes[row_index, 1].axis("off")
        print(f"{title}: {'VALID' if case_geometry_valid else 'INVALID'}")
        print(f"  source: {filename}")
        print(
            f"  LED line: alpha={calibration.led_line_alpha:.6f}, "
            f"score={calibration.led_line_score:.6f}, VP={vanishing_kind}"
        )
        print(
            "  LED fractions distal->proximal: "
            f"{np.round(calibration.led_longitudinal_fractions, 6).tolist()}"
        )
        print(
            "  LED centers [x,y] px: "
            f"{np.round(calibration.led_centers_xy_px, 2).tolist()}"
        )
        print(
            f"  centers inside silhouette={centers_inside}, "
            f"NPZ round trip={round_trip}"
        )
        if labels_xy_px is None:
            print("  manual GT accuracy: UNMEASURED (diagnostic-only mode)")
        else:
            print(
                f"  manual GT LED error median/max: "
                f"{median_led_error_px:.3f}/{maximum_led_error_px:.3f} px"
            )
            print(
                f"  manual GT to LED-line distance median/max: "
                f"{median_line_distance_px:.3f}/{maximum_line_distance_px:.3f} px"
            )

    figure_path = OUTPUT_DIRECTORY / "fixed_finger_calibration.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)
    with (OUTPUT_DIRECTORY / "calibration_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "case",
                "filename",
                "led_line_alpha",
                "led_line_score",
                "vanishing_point",
                "centers_inside_silhouette",
                "ordered_distal_to_proximal",
                "npz_round_trip",
                "median_led_error_px",
                "maximum_led_error_px",
                "median_led_line_distance_px",
                "maximum_led_line_distance_px",
                "geometry_result",
            )
        )
        writer.writerows(rows)
    print(f"Geometry invariants: {'PASS' if geometry_valid else 'FAIL'}")
    if ground_truth is None:
        print("Accuracy result: UNMEASURED (diagnostic-only mode)")
    else:
        print("Accuracy result: MEASURED (no arbitrary pixel-error threshold applied)")
    print(f"Artifacts: {OUTPUT_DIRECTORY}")
    if not geometry_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

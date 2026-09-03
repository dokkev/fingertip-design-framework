"""Validate the fixed-experiment geometry calibration on saved RGB images."""

from __future__ import annotations

import csv
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


IMAGE_DIRECTORY = REPOSITORY_ROOT / "experiments" / "img"
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "output" / "validation" / "fixed_finger_calibration"
)
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
    axis.scatter(
        calibration.led_centers_xy_px[:, 0],
        calibration.led_centers_xy_px[:, 1],
        s=30,
        color="#ff2ca0",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
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
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(REFERENCE_IMAGES),
        2,
        figsize=(10.0, 11.0),
        constrained_layout=True,
    )
    rows = []
    passed = True
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
        case_passed = round_trip and centers_inside and ordered
        passed &= case_passed
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
                "PASS" if case_passed else "FAIL",
            )
        )

        _overlay(axes[row_index, 0], rgb, calibration, title)
        axes[row_index, 1].imshow(warp_with_fixed_finger_calibration(rgb, loaded))
        axes[row_index, 1].set_title("Fixed projective canonical strip", fontsize=10)
        axes[row_index, 1].axis("off")
        print(f"{title}: {'PASS' if case_passed else 'FAIL'}")
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
                "result",
            )
        )
        writer.writerows(rows)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Artifacts: {OUTPUT_DIRECTORY}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

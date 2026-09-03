"""Validate the fixed-setup Solaris five-lobe localizer on saved images."""

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
    LedLocalizationResult,
    localize_solaris_leds,
    temporal_median_rgb,
)


IMAGE_DIRECTORY = REPOSITORY_ROOT / "experiments" / "img"
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "output" / "validation" / "solaris_led_localization"
)
GROUND_TRUTH_PATH = REPOSITORY_ROOT / "validation" / "solaris_led_ground_truth.json"
NORMAL_FILENAMES = tuple(f"solaris_p{index}_Color.png" for index in range(1, 7))
DARK_FILENAME = "solaris_unloaded_dark_Color.png"


def _load_rgb(filename: str) -> np.ndarray:
    bgr = cv2.imread(str(IMAGE_DIRECTORY / filename), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"missing Solaris image: {filename}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _label_ground_truth(
    normal_reference: np.ndarray, dark_reference: np.ndarray
) -> None:
    labels: dict[str, list[list[float]]] = {}
    for name, image in (
        ("normal_six_frame_median", normal_reference),
        (DARK_FILENAME, dark_reference),
    ):
        figure, axis = plt.subplots(figsize=(9.0, 6.0))
        axis.imshow(image)
        axis.set_title(f"{name}: click LED 1 to LED 5, distal to proximal")
        axis.axis("off")
        points = plt.ginput(5, timeout=-1, show_clicks=True)
        plt.close(figure)
        if len(points) != 5:
            raise RuntimeError(f"{name}: expected exactly five manual clicks")
        labels[name] = [[float(x), float(y)] for x, y in points]
    GROUND_TRUTH_PATH.write_text(
        json.dumps(
            {
                "coordinate_system": "original image pixels [x, y]",
                "point_order": "distal to proximal",
                "images": labels,
            },
            indent=2,
        )
        + "\n"
    )


def _load_ground_truth() -> dict[str, np.ndarray] | None:
    if not GROUND_TRUTH_PATH.is_file():
        return None
    payload = json.loads(GROUND_TRUTH_PATH.read_text())
    if payload.get("point_order") != "distal to proximal":
        raise ValueError("Solaris ground truth must be ordered distal to proximal")
    labels = {}
    for name in ("normal_six_frame_median", DARK_FILENAME):
        points = np.asarray(payload.get("images", {}).get(name), dtype=np.float64)
        if points.shape != (5, 2) or not np.all(np.isfinite(points)):
            raise ValueError(f"{name}: ground truth must be a finite 5 x 2 array")
        labels[name] = points
    return labels


def _draw_localization(
    axis: plt.Axes,
    rgb: np.ndarray,
    result: LedLocalizationResult,
    title: str,
    ground_truth: np.ndarray | None = None,
) -> None:
    axis.imshow(rgb)
    axis.contour(
        result.reference_mask,
        levels=(0.5,),
        colors=("#f1c40f",),
        linewidths=0.8,
    )
    centers = result.led_centers_xy_px
    axis.plot(
        centers[:, 0],
        centers[:, 1],
        color="#00a86b",
        linewidth=1.0,
        alpha=0.8,
    )
    axis.scatter(
        centers[:, 0],
        centers[:, 1],
        s=34,
        color="#ff2ca0",
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
        label="predicted",
    )
    if ground_truth is not None:
        axis.scatter(
            ground_truth[:, 0],
            ground_truth[:, 1],
            marker="x",
            s=44,
            color="#ffd166",
            linewidth=1.3,
            zorder=5,
            label="manual GT",
        )
        for predicted, expected in zip(centers, ground_truth, strict=True):
            axis.plot(
                (predicted[0], expected[0]),
                (predicted[1], expected[1]),
                color="#ffd166",
                linewidth=0.7,
                zorder=3,
            )
    for index, center in enumerate(centers, start=1):
        axis.text(
            center[0] + 4,
            center[1] - 3,
            str(index),
            color="white",
            fontsize=8,
            weight="bold",
            zorder=5,
        )
    axis.set_title(title, fontsize=10)
    if ground_truth is not None:
        axis.legend(loc="lower right", fontsize=7)
    axis.axis("off")


def _plot_profile(
    axis: plt.Axes,
    result: LedLocalizationResult,
    title: str,
) -> None:
    axis.plot(
        result.profile_rows_px,
        result.red_profile_dn,
        color="#999999",
        linewidth=1.0,
        label="raw red profile",
    )
    axis.plot(
        result.profile_rows_px,
        result.red_contrast_dn,
        color="#202020",
        linewidth=1.2,
        label="background-subtracted contrast",
    )
    for index, row in enumerate(result.peak_rows_px):
        axis.axvline(
            row,
            color="#ff2ca0",
            linestyle="--",
            linewidth=0.8,
            label="selected lobes" if index == 0 else None,
        )
    axis.set(
        xlabel="Image row, distal → proximal [px]",
        ylabel="Red-channel response [DN]",
        title=(
            f"{title}: {result.selected_side} side, score={result.sequence_score:.2f}"
        ),
    )
    axis.legend(loc="upper left", fontsize=8, frameon=False)


def _save_normal_overlay(
    frames: np.ndarray,
    result: LedLocalizationResult,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.8), constrained_layout=True)
    for index, (axis, frame) in enumerate(zip(axes.flat, frames, strict=True), start=1):
        _draw_localization(axis, frame, result, f"Normal frame {index}")
    figure.suptitle("One six-frame temporal-median calibration", fontsize=13)
    figure.savefig(OUTPUT_DIRECTORY / "normal_six_frame_overlay.png", dpi=200)
    figure.savefig(OUTPUT_DIRECTORY / "normal_six_frame_overlay.pdf")
    plt.close(figure)


def _save_reference_diagnostics(
    normal_reference: np.ndarray,
    normal_result: LedLocalizationResult,
    dark_reference: np.ndarray,
    dark_result: LedLocalizationResult,
    ground_truth: dict[str, np.ndarray] | None,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11.0, 7.2),
        constrained_layout=True,
    )
    normal_gt = (
        None if ground_truth is None else ground_truth["normal_six_frame_median"]
    )
    dark_gt = None if ground_truth is None else ground_truth[DARK_FILENAME]
    _draw_localization(
        axes[0, 0],
        normal_reference,
        normal_result,
        "Normal temporal median",
        normal_gt,
    )
    _plot_profile(axes[0, 1], normal_result, "Normal temporal median")
    _draw_localization(
        axes[1, 0],
        dark_reference,
        dark_result,
        "Dark-room reference",
        dark_gt,
    )
    _plot_profile(axes[1, 1], dark_result, "Dark-room reference")
    figure.savefig(
        OUTPUT_DIRECTORY / "reference_localization_and_profiles.png", dpi=200
    )
    figure.savefig(OUTPUT_DIRECTORY / "reference_localization_and_profiles.pdf")
    plt.close(figure)


def _terminal_stress_image(
    reference: np.ndarray,
    result: LedLocalizationResult,
) -> np.ndarray:
    stressed = reference.copy()
    pitch = float(np.median(np.diff(result.peak_rows_px)))
    first_terminal_row = int(round(result.peak_rows_px[-1] + pitch))
    for row in range(first_terminal_row, result.image_shape[0]):
        columns = np.flatnonzero(result.reference_mask[row])
        if len(columns) < 4:
            continue
        band_width = max(2, int(round(0.5 * len(columns))))
        selected = (
            columns[:band_width]
            if result.selected_side == "left"
            else columns[-band_width:]
        )
        stressed[row, selected] = 255
    return stressed


def _save_terminal_stress(
    reference: np.ndarray,
    original: LedLocalizationResult,
    stressed_image: np.ndarray,
    stressed: LedLocalizationResult,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    _draw_localization(axes[0], reference, original, "Original temporal median")
    _draw_localization(axes[1], stressed_image, stressed, "Saturated terminal leakage")
    figure.savefig(OUTPUT_DIRECTORY / "terminal_leakage_stress.png", dpi=200)
    figure.savefig(OUTPUT_DIRECTORY / "terminal_leakage_stress.pdf")
    plt.close(figure)


def _ground_truth_errors(
    result: LedLocalizationResult,
    ground_truth: np.ndarray,
) -> np.ndarray:
    return np.linalg.norm(result.led_centers_xy_px - ground_truth, axis=1)


def _write_csv(
    normal_result: LedLocalizationResult,
    dark_result: LedLocalizationResult,
    leave_one_out_shifts: np.ndarray,
    terminal_shifts: np.ndarray,
    ground_truth: dict[str, np.ndarray] | None,
) -> None:
    with (OUTPUT_DIRECTORY / "solaris_led_coordinates.csv").open(
        "w",
        newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("case", "led", "x_px", "y_px", "prominence_dn"))
        for name, result in (("normal", normal_result), ("dark", dark_result)):
            for index, (center, prominence) in enumerate(
                zip(
                    result.led_centers_xy_px,
                    result.peak_prominences_dn,
                    strict=True,
                ),
                start=1,
            ):
                writer.writerow((name, index, center[0], center[1], prominence))

    with (OUTPUT_DIRECTORY / "leave_one_frame_out.csv").open(
        "w",
        newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("held_out_frame", "led", "shift_px"))
        for frame_index, shifts in enumerate(leave_one_out_shifts, start=1):
            for led_index, shift in enumerate(shifts, start=1):
                writer.writerow((frame_index, led_index, shift))

    with (OUTPUT_DIRECTORY / "terminal_leakage_stress.csv").open(
        "w",
        newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("led", "shift_px"))
        for led_index, shift in enumerate(terminal_shifts, start=1):
            writer.writerow((led_index, shift))

    with (OUTPUT_DIRECTORY / "solaris_led_ground_truth_errors.csv").open(
        "w",
        newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("case", "led", "euclidean_error_px"))
        if ground_truth is not None:
            for name, result, key in (
                ("normal", normal_result, "normal_six_frame_median"),
                ("dark", dark_result, DARK_FILENAME),
            ):
                for led_index, error in enumerate(
                    _ground_truth_errors(result, ground_truth[key]),
                    start=1,
                ):
                    writer.writerow((name, led_index, error))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-ground-truth",
        action="store_true",
        help="click five Solaris LED centers distal to proximal",
    )
    arguments = parser.parse_args()

    normal_frames = np.stack([_load_rgb(name) for name in NORMAL_FILENAMES])
    normal_reference = temporal_median_rgb(normal_frames)
    dark_reference = _load_rgb(DARK_FILENAME)
    if arguments.label_ground_truth:
        _label_ground_truth(normal_reference, dark_reference)
    ground_truth = _load_ground_truth()

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    normal_result = localize_solaris_leds(normal_frames)
    dark_result = localize_solaris_leds(dark_reference)

    leave_one_out_results = [
        localize_solaris_leds(np.delete(normal_frames, held_out, axis=0))
        for held_out in range(len(normal_frames))
    ]
    leave_one_out_shifts = np.vstack(
        [
            np.linalg.norm(
                result.led_centers_xy_px - normal_result.led_centers_xy_px,
                axis=1,
            )
            for result in leave_one_out_results
        ]
    )

    stressed_image = _terminal_stress_image(normal_reference, normal_result)
    # Geometry is fixed in this ablation: reuse the original silhouette so the
    # comparison changes terminal photometry, not GrabCut support.
    stressed_result = localize_solaris_leds(
        stressed_image,
        reference_mask=normal_result.reference_mask,
    )
    terminal_shifts = np.linalg.norm(
        stressed_result.led_centers_xy_px - normal_result.led_centers_xy_px,
        axis=1,
    )

    _save_normal_overlay(normal_frames, normal_result)
    _save_reference_diagnostics(
        normal_reference,
        normal_result,
        dark_reference,
        dark_result,
        ground_truth,
    )
    _save_terminal_stress(
        normal_reference,
        normal_result,
        stressed_image,
        stressed_result,
    )
    _write_csv(
        normal_result,
        dark_result,
        leave_one_out_shifts,
        terminal_shifts,
        ground_truth,
    )

    print("Normal six-frame temporal-median localization")
    print(f"  selected side: {normal_result.selected_side}")
    print(
        f"  LED centers [x,y] px: {np.round(normal_result.led_centers_xy_px, 3).tolist()}"
    )
    print(f"  peak rows [px]: {normal_result.peak_rows_px.tolist()}")
    print(
        f"  prominences [DN]: {np.round(normal_result.peak_prominences_dn, 3).tolist()}"
    )
    print(f"  sequence score: {normal_result.sequence_score:.6f}")
    print("Dark-room localization")
    print(f"  selected side: {dark_result.selected_side}")
    print(
        f"  LED centers [x,y] px: {np.round(dark_result.led_centers_xy_px, 3).tolist()}"
    )
    print(f"  peak rows [px]: {dark_result.peak_rows_px.tolist()}")
    print(
        f"  prominences [DN]: {np.round(dark_result.peak_prominences_dn, 3).tolist()}"
    )
    print(f"  sequence score: {dark_result.sequence_score:.6f}")
    print(
        "Leave-one-frame-out LED shift median/max [px]: "
        f"{np.median(leave_one_out_shifts):.6f}/"
        f"{np.max(leave_one_out_shifts):.6f}"
    )
    print(
        "Terminal-leakage stress LED shift median/max [px]: "
        f"{np.median(terminal_shifts):.6f}/{np.max(terminal_shifts):.6f}"
    )
    if ground_truth is None:
        print("Manual GT: UNAVAILABLE")
    else:
        for name, result, key in (
            ("normal", normal_result, "normal_six_frame_median"),
            ("dark", dark_result, DARK_FILENAME),
        ):
            errors = _ground_truth_errors(result, ground_truth[key])
            print(
                f"Manual GT {name} per-LED error [px]: {np.round(errors, 3).tolist()}"
            )
            print(
                f"Manual GT {name} median/max [px]: "
                f"{np.median(errors):.6f}/{np.max(errors):.6f}"
            )
    print(f"Artifacts: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()

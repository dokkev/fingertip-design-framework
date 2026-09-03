"""Validate the offline Solaris periodic-array LED localizer on saved images."""

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
)
from experiments.localization.solaris_led_localization import (  # noqa: E402
    PERIODIC_PROFILE_SAMPLE_COUNT,
    _periodic_responses,
    _sample_periodic_profiles,
    solaris_physical_led_layout,
)


IMAGE_DIRECTORY = REPOSITORY_ROOT / "experiments" / "img"
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "output" / "validation" / "solaris_led_localization"
)
GROUND_TRUTH_PATH = REPOSITORY_ROOT / "validation" / "solaris_led_ground_truth.json"
REFERENCE_IMAGES = (
    ("Solaris representative", "solaris_p1_Color.png"),
    ("Dark-room Solaris", "solaris_unloaded_dark_Color.png"),
)


def _load_rgb(filename: str) -> np.ndarray:
    bgr = cv2.imread(str(IMAGE_DIRECTORY / filename), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"missing Solaris image: {filename}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _label_ground_truth() -> None:
    labels: dict[str, list[list[float]]] = {}
    for title, filename in REFERENCE_IMAGES:
        rgb = _load_rgb(filename)
        figure, axis = plt.subplots(figsize=(12.0, 7.0))
        axis.imshow(rgb)
        axis.set_title(f"{title}: click LED 1 to LED 5, distal to proximal")
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
    print(f"Manual Solaris LED ground truth: {GROUND_TRUTH_PATH}")


def _load_ground_truth() -> dict[str, np.ndarray] | None:
    if not GROUND_TRUTH_PATH.is_file():
        return None
    payload = json.loads(GROUND_TRUTH_PATH.read_text())
    if payload.get("point_order") != "distal to proximal":
        raise ValueError("Solaris manual labels must be ordered distal to proximal")
    labels = {}
    for _, filename in REFERENCE_IMAGES:
        points = np.asarray(payload.get("images", {}).get(filename), dtype=np.float64)
        if points.shape != (5, 2) or not np.all(np.isfinite(points)):
            raise ValueError(f"{filename}: manual labels must be a finite 5 x 2 array")
        labels[filename] = points
    return labels


def _line_segment(result: LedLocalizationResult, line: np.ndarray) -> np.ndarray:
    height, width = result.image_shape
    points = []
    if abs(line[1]) > np.finfo(np.float64).eps:
        for x in (0.0, float(width - 1)):
            y = -(line[0] * x + line[2]) / line[1]
            if 0.0 <= y < height:
                points.append((x, y))
    if abs(line[0]) > np.finfo(np.float64).eps:
        for y in (0.0, float(height - 1)):
            x = -(line[1] * y + line[2]) / line[0]
            if 0.0 <= x < width:
                points.append((x, y))
    unique = np.unique(np.round(np.asarray(points), 9), axis=0)
    if len(unique) < 2:
        raise RuntimeError("image line does not cross two image boundaries")
    return unique[:2]


def _transverse_segment(
    result: LedLocalizationResult,
    limit: np.ndarray,
) -> np.ndarray:
    points = []
    for side in (result.dorsal_line, result.palmar_line):
        point_h = np.cross(side, limit)
        points.append(point_h[:2] / point_h[2])
    return np.asarray(points)


def _selected_profile(
    rgb: np.ndarray,
    result: LedLocalizationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    led_positions_mm = solaris_physical_led_layout()
    distances_mm = np.linspace(
        led_positions_mm[0],
        led_positions_mm[-1],
        PERIODIC_PROFILE_SAMPLE_COUNT,
    )
    profiles, valid = _sample_periodic_profiles(
        rgb[:, :, 0].astype(np.float32),
        result.led_line,
        result.distal_limit,
        result.vanishing_point_h,
        np.asarray((result.longitudinal_scale_px_per_mm,)),
        distances_mm,
    )
    contrast, center_responses, midpoint_responses, scores = _periodic_responses(
        distances_mm,
        profiles,
        valid,
        led_positions_mm,
    )
    return (
        distances_mm,
        profiles[0],
        contrast[0],
        center_responses[0],
        midpoint_responses[0],
        float(scores[0]),
    )


def _overlay(
    axis: plt.Axes,
    rgb: np.ndarray,
    result: LedLocalizationResult,
    ground_truth: np.ndarray | None,
    title: str,
) -> None:
    axis.imshow(rgb)
    axis.contour(
        result.reference_mask,
        levels=(0.5,),
        colors=("#f1c40f",),
        linewidths=1.2,
    )
    for line, color, label in (
        (result.dorsal_line, "#31688e", "dorsal"),
        (result.palmar_line, "#e67e22", "palmar"),
        (result.led_line, "#00a86b", "LED line"),
    ):
        segment = _line_segment(result, line)
        axis.plot(segment[:, 0], segment[:, 1], color=color, linewidth=1.8, label=label)
    distal_segment = _transverse_segment(result, result.distal_limit)
    axis.plot(
        distal_segment[:, 0],
        distal_segment[:, 1],
        color="#6c757d",
        linewidth=1.4,
        linestyle="--",
        label="distal end",
    )
    axis.scatter(
        result.led_centers_xy_px[:, 0],
        result.led_centers_xy_px[:, 1],
        s=34,
        color="#ff2ca0",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="rigid-array LEDs",
    )
    normal = result.led_line[:2].copy()
    array_midpoint = np.mean(result.led_centers_xy_px[[0, 4]], axis=0)
    mask_y, mask_x = np.nonzero(result.reference_mask)
    mask_centroid = np.asarray((np.mean(mask_x), np.mean(mask_y)))
    if np.dot(normal, array_midpoint - mask_centroid) < 0.0:
        normal *= -1.0
    bracket = result.led_centers_xy_px[[0, 4]] + 24.0 * normal
    axis.annotate(
        "",
        xy=bracket[1],
        xytext=bracket[0],
        arrowprops={"arrowstyle": "|-|", "color": "white", "linewidth": 1.2},
        zorder=5,
    )
    axis.text(
        0.03,
        0.94,
        "Rigid array: 11 mm pitch, 44 mm span",
        transform=axis.transAxes,
        color="white",
        fontsize=8,
        ha="left",
        va="top",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.45, "pad": 2},
        zorder=5,
    )
    if ground_truth is not None:
        axis.scatter(
            ground_truth[:, 0],
            ground_truth[:, 1],
            marker="x",
            s=50,
            color="#ffd166",
            linewidth=1.4,
            zorder=5,
            label="manual GT",
        )
    for index, center in enumerate(result.led_centers_xy_px, start=1):
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
        f"{title} — Full 11 mm periodic array match\n"
        f"$\\alpha$={result.led_line_alpha:.3f}, "
        f"scale={result.longitudinal_scale_px_per_mm:.3f} px/mm, "
        f"score={result.line_score:.3f} DN",
        fontsize=10,
    )
    axis.legend(loc="lower right", fontsize=7, framealpha=0.9)
    axis.axis("off")


def _profile_plot(
    axis: plt.Axes,
    distances_mm: np.ndarray,
    profile: np.ndarray,
    contrast: np.ndarray,
    center_responses: np.ndarray,
    midpoint_responses: np.ndarray,
    score: float,
) -> None:
    led_positions_mm = solaris_physical_led_layout()
    midpoint_positions_mm = 0.5 * (led_positions_mm[:-1] + led_positions_mm[1:])
    axis.axvspan(
        led_positions_mm[0],
        led_positions_mm[-1],
        color="#e8f5e9",
        alpha=0.35,
        linewidth=0.0,
        label="scored LED1–LED5 interval",
    )
    for index, center_mm in enumerate(led_positions_mm, start=1):
        axis.axvline(
            center_mm,
            color="#00a86b",
            linestyle="--",
            linewidth=0.9,
            label="LED centers" if index == 1 else None,
        )
        axis.scatter(
            center_mm,
            center_responses[index - 1],
            color="#00a86b",
            edgecolor="white",
            linewidth=0.5,
            s=30,
            zorder=4,
        )
    for index, midpoint_mm in enumerate(midpoint_positions_mm):
        axis.axvline(
            midpoint_mm,
            color="#7b2cbf",
            linestyle=":",
            linewidth=0.9,
            label="inter-LED midpoints" if index == 0 else None,
        )
        axis.scatter(
            midpoint_mm,
            midpoint_responses[index],
            marker="v",
            color="#7b2cbf",
            edgecolor="white",
            linewidth=0.5,
            s=30,
            zorder=4,
        )
    axis.plot(
        distances_mm,
        profile,
        color="#9e9e9e",
        linewidth=0.8,
        label="raw red profile",
    )
    axis.plot(
        distances_mm,
        contrast,
        color="#303030",
        linewidth=1.2,
        label="signed red contrast",
    )
    axis.set(
        xlim=(float(led_positions_mm[0]), float(led_positions_mm[-1])),
        xlabel="Distance from distal end [mm]",
        ylabel="Red-channel value [DN]",
        title=f"Full 11 mm periodic array match: score={score:.3f} DN",
    )
    axis.tick_params(labelsize=8)
    axis.legend(loc="upper left", fontsize=7, frameon=False)


def _ground_truth_errors(
    result: LedLocalizationResult,
    ground_truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    error_vectors = result.led_centers_xy_px - ground_truth
    normal = result.led_line[:2]
    tangent = np.asarray((-normal[1], normal[0]))
    euclidean = np.linalg.norm(error_vectors, axis=1)
    longitudinal = np.abs(error_vectors @ tangent)
    transverse = np.abs(error_vectors @ normal)
    return euclidean, longitudinal, transverse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-ground-truth",
        action="store_true",
        help="click five Solaris LED centers distal to proximal",
    )
    arguments = parser.parse_args()
    if arguments.label_ground_truth:
        _label_ground_truth()
    ground_truth = _load_ground_truth()

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(REFERENCE_IMAGES),
        2,
        figsize=(11.0, 7.2),
        gridspec_kw={"width_ratios": (1.25, 1.0)},
        constrained_layout=True,
    )
    rows = []
    error_rows = []
    for row_index, (title, filename) in enumerate(REFERENCE_IMAGES):
        rgb = _load_rgb(filename)
        cv2.setRNGSeed(0)
        result = localize_solaris_leds(rgb)
        (
            distances_mm,
            profile,
            contrast,
            center_responses,
            midpoint_responses,
            score,
        ) = _selected_profile(rgb, result)
        if not np.allclose(center_responses, result.led_center_responses):
            raise RuntimeError("recomputed Solaris center responses do not match")
        if not np.allclose(midpoint_responses, result.inter_led_responses):
            raise RuntimeError("recomputed Solaris midpoint responses do not match")
        if not np.isclose(score, result.line_score):
            raise RuntimeError("recomputed Solaris periodic score does not match")
        labels = None if ground_truth is None else ground_truth[filename]
        if labels is None:
            euclidean = longitudinal = transverse = np.full(5, np.nan)
        else:
            euclidean, longitudinal, transverse = _ground_truth_errors(result, labels)
            for led_index in range(5):
                error_rows.append(
                    (
                        title,
                        filename,
                        led_index + 1,
                        euclidean[led_index],
                        longitudinal[led_index],
                        transverse[led_index],
                    )
                )

        _overlay(axes[row_index, 0], rgb, result, labels, title)
        _profile_plot(
            axes[row_index, 1],
            distances_mm,
            profile,
            contrast,
            center_responses,
            midpoint_responses,
            score,
        )
        rows.append(
            (
                title,
                filename,
                result.led_line_alpha,
                result.longitudinal_scale_px_per_mm,
                result.line_score,
                *result.led_center_responses,
                *result.inter_led_responses,
                float(np.nanmedian(euclidean)) if labels is not None else "",
                float(np.nanmax(euclidean)) if labels is not None else "",
                float(np.nanmedian(longitudinal)) if labels is not None else "",
                float(np.nanmax(longitudinal)) if labels is not None else "",
                float(np.nanmedian(transverse)) if labels is not None else "",
                float(np.nanmax(transverse)) if labels is not None else "",
            )
        )
        print(title)
        print(f"  source: {filename}")
        print(
            f"  LED line alpha={result.led_line_alpha:.6f}, "
            f"distal scale={result.longitudinal_scale_px_per_mm:.6f} px/mm, "
            f"periodic score={result.line_score:.6f} DN"
        )
        print(
            "  LED-center responses [DN]: "
            f"{np.round(result.led_center_responses, 6).tolist()}"
        )
        print(
            "  inter-LED midpoint responses [DN]: "
            f"{np.round(result.inter_led_responses, 6).tolist()}"
        )
        print(
            "  physical positions from distal [mm]: "
            f"{result.longitudinal_positions_mm.tolist()}"
        )
        print(
            "  predicted centers [x,y] px: "
            f"{np.round(result.led_centers_xy_px, 2).tolist()}"
        )
        if labels is None:
            print("  manual GT accuracy: UNMEASURED")
        else:
            print(
                "  LED error median/max [px]: "
                f"{np.median(euclidean):.3f}/{np.max(euclidean):.3f}"
            )
            print(
                "  longitudinal error median/max [px]: "
                f"{np.median(longitudinal):.3f}/{np.max(longitudinal):.3f}"
            )
            print(
                "  transverse error median/max [px]: "
                f"{np.median(transverse):.3f}/{np.max(transverse):.3f}"
            )
            print(f"  per-LED Euclidean error [px]: {np.round(euclidean, 3).tolist()}")

    figure.savefig(OUTPUT_DIRECTORY / "solaris_led_localization.png", dpi=200)
    figure.savefig(OUTPUT_DIRECTORY / "solaris_led_localization.pdf")
    plt.close(figure)
    with (OUTPUT_DIRECTORY / "solaris_led_localization.csv").open(
        "w",
        newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "case",
                "filename",
                "led_line_alpha",
                "longitudinal_scale_px_per_mm",
                "periodic_match_score_dn",
                "led1_center_response_dn",
                "led2_center_response_dn",
                "led3_center_response_dn",
                "led4_center_response_dn",
                "led5_center_response_dn",
                "midpoint12_response_dn",
                "midpoint23_response_dn",
                "midpoint34_response_dn",
                "midpoint45_response_dn",
                "median_led_error_px",
                "maximum_led_error_px",
                "median_longitudinal_error_px",
                "maximum_longitudinal_error_px",
                "median_transverse_error_px",
                "maximum_transverse_error_px",
            )
        )
        writer.writerows(rows)
    with (OUTPUT_DIRECTORY / "solaris_led_ground_truth_errors.csv").open(
        "w",
        newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "case",
                "filename",
                "led_index_distal_to_proximal",
                "euclidean_error_px",
                "longitudinal_error_px",
                "transverse_error_px",
            )
        )
        writer.writerows(error_rows)
    print(
        "Accuracy result: "
        + ("MEASURED (no pass threshold)" if ground_truth is not None else "UNMEASURED")
    )
    print(f"Artifacts: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()

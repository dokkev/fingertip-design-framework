"""Plot the brightest-10% red response for the 5 mm contact sweep."""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lumo.visualization import (  # noqa: E402
    DEFAULT_STYLE,
    publication_context,
    save_figure,
)


_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIRECTORY = _ROOT / "output" / "experiments"
OUTPUT_DIRECTORY = Path(__file__).resolve().parent

CONTACT_SPACING_MM = 5.0
LED_SPACING_MM = 11.0
CONTACT_COUNT = 7
LED_COUNT = 5
TOP_FRACTION = 0.10
SHOW_TITLE = True

ROI_WIDTH_IN_LED_SPACINGS = 1.70
ROI_HEIGHT_IN_LED_SPACINGS = 0.76
ROI_INWARD_SHIFT_IN_LED_SPACINGS = 0.35

SMALL_GAUSSIAN_SIGMA_PX = 1.2
BROAD_GAUSSIAN_SIGMA_PX = 14.0
SEARCH_X_FRACTION = (0.50, 0.70)
SEARCH_Y_FRACTION = (0.28, 0.62)


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"OpenCV could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _contact_frames() -> tuple[tuple[Path, ...], np.ndarray]:
    indexed: dict[int, Path] = {}
    for path in IMAGE_DIRECTORY.glob("*.png"):
        match = re.fullmatch(r"p(\d+)_Color\.png", path.name, flags=re.IGNORECASE)
        if match:
            indexed[int(match.group(1))] = path
    expected = tuple(range(CONTACT_COUNT))
    if tuple(sorted(indexed)) != expected:
        raise RuntimeError(
            f"expected p0-p{CONTACT_COUNT - 1}_Color.png, found "
            f"{[indexed[index].name for index in sorted(indexed)]}"
        )
    paths = tuple(indexed[index] for index in expected)
    positions_mm = CONTACT_SPACING_MM * np.arange(CONTACT_COUNT, dtype=np.float64)
    return paths, positions_mm


def _unloaded_frame(contact_paths: tuple[Path, ...]) -> Path | None:
    contact_set = {path.resolve() for path in contact_paths}
    tokens = ("unloaded", "no_contact", "no-contact", "nocontact", "baseline")
    matches = [
        path
        for path in IMAGE_DIRECTORY.glob("*.png")
        if path.resolve() not in contact_set
        and any(token in path.stem.lower() for token in tokens)
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple possible unloaded frames: {matches}")
    return matches[0] if matches else None


def _red_high_pass(rgb: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.float32)
    small = cv2.GaussianBlur(red, (0, 0), SMALL_GAUSSIAN_SIGMA_PX)
    background = cv2.GaussianBlur(red, (0, 0), BROAD_GAUSSIAN_SIGMA_PX)
    return np.maximum(small - background, 0.0)


def _smooth_profile(values: np.ndarray, sigma_px: float) -> np.ndarray:
    return cv2.GaussianBlur(
        np.asarray(values, dtype=np.float32)[:, None],
        (1, 0),
        sigma_px,
    ).ravel()


def _candidate_peaks(
    profile: np.ndarray,
    start: int,
    stop: int,
    minimum_distance: int,
) -> np.ndarray:
    local_span = float(np.ptp(profile[start:stop]))
    minimum_prominence = 0.018 * local_span
    candidates = []
    for index in range(start + 1, stop - 1):
        if profile[index] < profile[index - 1] or profile[index] <= profile[index + 1]:
            continue
        left = max(start, index - minimum_distance)
        right = min(stop, index + minimum_distance + 1)
        local_floor = max(
            float(np.min(profile[left : index + 1])),
            float(np.min(profile[index:right])),
        )
        if profile[index] - local_floor >= minimum_prominence:
            candidates.append(index)

    kept: list[int] = []
    for index in sorted(
        candidates, key=lambda item: float(profile[item]), reverse=True
    ):
        if all(abs(index - selected) >= minimum_distance for selected in kept):
            kept.append(index)
    return np.asarray(sorted(kept), dtype=np.int32)


def _detect_led_landmarks(rgb_images: np.ndarray) -> np.ndarray:
    """Detect one common five-LED array from the fixed-camera median frame."""

    median_rgb = np.median(rgb_images, axis=0).astype(np.uint8)
    high_pass = _red_high_pass(median_rgb)
    height, width = high_pass.shape
    x_start = round(SEARCH_X_FRACTION[0] * width)
    x_stop = round(SEARCH_X_FRACTION[1] * width)
    y_start = round(SEARCH_Y_FRACTION[0] * height)
    y_stop = round(SEARCH_Y_FRACTION[1] * height)
    profile = _smooth_profile(high_pass[:, x_start:x_stop].mean(axis=1), 1.4)
    peak_y = _candidate_peaks(
        profile,
        y_start,
        y_stop,
        minimum_distance=round(0.018 * height),
    )
    if peak_y.size < LED_COUNT:
        raise RuntimeError(f"red detector found only {peak_y.size} candidate peaks")

    peak_x = []
    strengths = []
    row_half_width = round(0.012 * height)
    for y_coordinate in peak_y:
        column_profile = _smooth_profile(
            high_pass[
                max(0, y_coordinate - row_half_width) : min(
                    height, y_coordinate + row_half_width + 1
                ),
                x_start:x_stop,
            ].sum(axis=0),
            1.5,
        )
        peak_x.append(x_start + int(np.argmax(column_profile)))
        strengths.append(float(profile[y_coordinate]))

    peak_x_array = np.asarray(peak_x, dtype=np.float64)
    strength_array = np.asarray(strengths, dtype=np.float64)
    best: tuple[float, tuple[int, ...]] | None = None
    for selection in itertools.combinations(range(peak_y.size), LED_COUNT):
        selected_y = peak_y[list(selection)].astype(np.float64)
        selected_x = peak_x_array[list(selection)]
        spacing = np.diff(selected_y)
        mean_spacing = float(np.mean(spacing))
        if not 0.025 * height <= mean_spacing <= 0.065 * height:
            continue
        if float(np.min(spacing)) < 0.014 * height:
            continue
        spacing_cv = float(np.std(spacing) / mean_spacing)
        line = np.polyfit(selected_y, selected_x, 1)
        line_rmse = float(
            np.sqrt(np.mean((selected_x - np.polyval(line, selected_y)) ** 2))
        )
        score = (
            float(np.mean(strength_array[list(selection)]))
            / (float(np.max(strength_array)) + 1.0e-6)
            - 2.0 * spacing_cv
            - line_rmse / (0.5 * mean_spacing)
        )
        if best is None or score > best[0]:
            best = (score, selection)
    if best is None:
        raise RuntimeError("red detector could not select five regular ordered peaks")

    selection = list(best[1])
    landmarks = np.column_stack((peak_x_array[selection], peak_y[selection])).astype(
        np.float64
    )
    spacings = np.linalg.norm(np.diff(landmarks, axis=0), axis=1)
    spacing_cv = float(np.std(spacings) / np.mean(spacings))
    if landmarks.shape != (LED_COUNT, 2) or not np.all(np.diff(landmarks[:, 1]) > 0):
        raise RuntimeError("LED landmarks are not five ordered image points")
    if spacing_cv > 0.30:
        raise RuntimeError(f"detected LED spacing is not regular: CV={spacing_cv:.3f}")
    return landmarks


def _roi_polygons(landmarks: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    spacing_px = float(np.median(np.linalg.norm(np.diff(landmarks, axis=0), axis=1)))
    array_axis = landmarks[-1] - landmarks[0]
    array_axis /= np.linalg.norm(array_axis)
    outward_axis = np.asarray((array_axis[1], -array_axis[0]))
    if outward_axis[0] < 0.0:
        outward_axis *= -1.0

    half_width = 0.5 * ROI_WIDTH_IN_LED_SPACINGS * spacing_px
    half_height = 0.5 * ROI_HEIGHT_IN_LED_SPACINGS * spacing_px
    centers = landmarks - ROI_INWARD_SHIFT_IN_LED_SPACINGS * spacing_px * outward_axis
    polygons = []
    for center in centers:
        polygons.append(
            (
                center - half_width * outward_axis - half_height * array_axis,
                center + half_width * outward_axis - half_height * array_axis,
                center + half_width * outward_axis + half_height * array_axis,
                center - half_width * outward_axis + half_height * array_axis,
            )
        )
    polygon_array = np.asarray(polygons, dtype=np.float64)
    height, width = image_shape[:2]
    if (
        np.min(polygon_array[:, :, 0]) < 0.0
        or np.max(polygon_array[:, :, 0]) >= width
        or np.min(polygon_array[:, :, 1]) < 0.0
        or np.max(polygon_array[:, :, 1]) >= height
    ):
        raise RuntimeError("a detected LED ROI extends outside the image")
    return polygon_array


def _brightest_red_features(rgb: np.ndarray, polygons: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.float64)
    values = []
    for polygon in polygons:
        mask = np.zeros(red.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
        pixels = red[mask > 0]
        if pixels.size == 0:
            raise RuntimeError("an LED ROI contains no pixels")
        count = max(1, int(np.ceil(TOP_FRACTION * pixels.size)))
        values.append(
            float(np.mean(np.partition(pixels, pixels.size - count)[-count:]))
        )
    return np.asarray(values, dtype=np.float64)


def _save_debug_overlay(
    paths: tuple[Path, ...],
    rgb_images: np.ndarray,
    landmarks: np.ndarray,
    polygons: np.ndarray,
) -> Path:
    panels = []
    for path, rgb in zip(paths, rgb_images, strict=True):
        panel = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for index, (landmark, polygon) in enumerate(
            zip(landmarks, polygons, strict=True), start=1
        ):
            cv2.polylines(
                panel,
                [np.rint(polygon).astype(np.int32)],
                True,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            center = tuple(np.rint(landmark).astype(int))
            cv2.circle(panel, center, 4, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                panel,
                f"LED{index}",
                (center[0] + 5, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            panel,
            path.stem,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)

    panels.append(np.full_like(panels[0], 255))
    montage = np.concatenate(
        [
            np.concatenate(panels[0:4], axis=1),
            np.concatenate(panels[4:8], axis=1),
        ],
        axis=0,
    )
    output = OUTPUT_DIRECTORY / "brightest10_red_contact_sweep_rois_debug.png"
    if not cv2.imwrite(str(output), montage):
        raise RuntimeError(f"failed to write {output}")
    return output


def _plot_heatmap(
    response: np.ndarray,
    contact_positions_mm: np.ndarray,
    nearest_led: np.ndarray,
    *,
    median_centered: bool,
) -> tuple[Path, ...]:
    with publication_context():
        figure, axes = plt.subplots(figsize=(3.50, 3.20), constrained_layout=True)
        image = axes.imshow(response, cmap="viridis", aspect="auto")
        axes.set_xticks(np.arange(LED_COUNT), [f"LED {index}" for index in range(1, 6)])
        axes.set_yticks(
            np.arange(CONTACT_COUNT),
            [
                f"p{index} ({position:g} mm)"
                for index, position in enumerate(contact_positions_mm)
            ],
        )
        axes.set_xlabel("LED-centered region")
        axes.set_ylabel("Contact position (distal to proximal)")
        marker = axes.scatter(
            nearest_led,
            np.arange(CONTACT_COUNT),
            marker="x",
            s=28,
            linewidth=1.0,
            color="white",
            label="Nearest physical LED",
        )
        marker.set_path_effects(
            [
                path_effects.Stroke(linewidth=1.8, foreground="#333333"),
                path_effects.Normal(),
            ]
        )
        axes.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            frameon=False,
            fontsize=6.4,
            handletextpad=0.3,
        )
        if SHOW_TITLE:
            title = "Brightest-10% red response across 5 mm contact sweep"
            if median_centered:
                title += "\nmedian-centered / exploratory"
            axes.set_title(title, pad=25.0)
        colorbar = figure.colorbar(image, ax=axes, pad=0.025)
        colorbar.set_label(
            "Median-centered red response [8-bit DN]"
            if median_centered
            else r"$\Delta C_i$ from unloaded [8-bit DN]"
        )
        stem = OUTPUT_DIRECTORY / "brightest10_red_contact_sweep"
        if median_centered:
            stem = stem.with_name(stem.name + "_median_centered")
        outputs = save_figure(
            figure,
            stem,
            formats=("png", "pdf"),
            style=DEFAULT_STYLE,
        )
        plt.close(figure)
    return outputs


def main() -> None:
    contact_paths, contact_positions_mm = _contact_frames()
    rgb_images = np.stack([_read_rgb(path) for path in contact_paths])
    if len({image.shape for image in rgb_images}) != 1:
        raise RuntimeError("contact frames do not share one image shape")

    unloaded_path = _unloaded_frame(contact_paths)
    landmarks = _detect_led_landmarks(rgb_images)
    polygons = _roi_polygons(landmarks, rgb_images.shape[1:])
    features = np.stack(
        [_brightest_red_features(image, polygons) for image in rgb_images]
    )

    if unloaded_path is None:
        print(
            "WARNING: no unloaded/no-contact frame was found; using a "
            "median-centered exploratory response."
        )
        response = features - np.median(features, axis=0, keepdims=True)
        median_centered = True
        baseline_name = "none"
    else:
        unloaded = _read_rgb(unloaded_path)
        if unloaded.shape != rgb_images.shape[1:]:
            raise RuntimeError("unloaded frame shape differs from the contact frames")
        baseline = _brightest_red_features(unloaded, polygons)
        response = features - baseline
        normalized = response / np.maximum(baseline, np.finfo(np.float64).eps)
        print(
            "baseline comparison: absolute range="
            f"[{response.min():+.6f}, {response.max():+.6f}], normalized range="
            f"[{normalized.min():+.6f}, {normalized.max():+.6f}]"
        )
        median_centered = False
        baseline_name = unloaded_path.name

    led_positions_mm = LED_SPACING_MM * np.arange(LED_COUNT, dtype=np.float64)
    nearest_led = np.argmin(
        np.abs(contact_positions_mm[:, None] - led_positions_mm[None, :]), axis=1
    )
    predicted_led = np.argmax(response, axis=1)
    sorted_response = np.sort(response, axis=1)
    margins = sorted_response[:, -1] - sorted_response[:, -2]

    spacing_px = np.linalg.norm(np.diff(landmarks, axis=0), axis=1)
    print(f"loaded contact frames: {[path.name for path in contact_paths]}")
    print(f"baseline frame: {baseline_name}")
    print(f"LED landmarks [x, y] px: {np.round(landmarks, 2).tolist()}")
    print(f"median LED spacing: {np.median(spacing_px):.3f} px")
    print(
        "ROI rule: width="
        f"{ROI_WIDTH_IN_LED_SPACINGS:.2f}x spacing, height="
        f"{ROI_HEIGHT_IN_LED_SPACINGS:.2f}x spacing, inward shift="
        f"{ROI_INWARD_SHIFT_IN_LED_SPACINGS:.2f}x spacing"
    )
    print("contact_mm  nearest_LED  predicted_LED  top2_margin")
    for position, nearest, predicted, margin in zip(
        contact_positions_mm, nearest_led, predicted_led, margins, strict=True
    ):
        print(
            f"{position:10.1f}  {nearest + 1:11d}  {predicted + 1:13d}  {margin:11.6f}"
        )
    print(f"nearest-LED accuracy: {np.mean(nearest_led == predicted_led):.1%}")
    print(
        "predicted LED sequence: "
        f"{(predicted_led + 1).tolist()} "
        f"({'monotonic' if np.all(np.diff(predicted_led) >= 0) else 'not monotonic'})"
    )

    debug_output = _save_debug_overlay(contact_paths, rgb_images, landmarks, polygons)
    figure_outputs = _plot_heatmap(
        response,
        contact_positions_mm,
        nearest_led,
        median_centered=median_centered,
    )
    print(f"wrote {debug_output}")
    for output in figure_outputs:
        print(f"wrote {output}")


if __name__ == "__main__":
    main()

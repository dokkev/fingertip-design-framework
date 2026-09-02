"""Replay the paired-LSD fingertip boundary on the 13 reference images."""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.localization import detect_fingertip_boundary  # noqa: E402


IMAGE_DIRECTORY = REPOSITORY_ROOT / "experiments" / "img"
OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "output"
    / "validation"
    / "fingertip_boundary_lsd"
    / "paired_lsd_13_image_validation.png"
)

# Manually accepted central boundary positions, normalized by image width.
# The tolerance rejects the previously observed right-fin false positive while
# allowing small line-fit changes from OpenCV and image downsampling.
EXPECTED_CENTER_BOUNDARIES = {
    "p0_Color.png": (0.4803, 0.6471),
    "p1_Color.png": (0.4876, 0.6396),
    "p2_Color.png": (0.4983, 0.6271),
    "p3_Color.png": (0.4867, 0.6399),
    "p4_Color.png": (0.4926, 0.6369),
    "p5_Color.png": (0.4925, 0.6371),
    "p6_Color.png": (0.4917, 0.6369),
    "p0d_Color.png": (0.5145, 0.6072),
    "p1d_Color.png": (0.5124, 0.6041),
    "p2d_Color.png": (0.5126, 0.6013),
    "p3d_Color.png": (0.5129, 0.6015),
    "p4d_Color.png": (0.5131, 0.6052),
    "p5d_Color.png": (0.5132, 0.6060),
}
MAXIMUM_CENTER_ERROR_WIDTH_FRACTION = 0.025


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"could not read reference image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _center_boundaries(region, image_width: int) -> tuple[float, float]:
    core_center_y = 0.5 * sum(region.core_y_span)
    dorsal_x = float(
        np.interp(
            core_center_y,
            region.dorsal_boundary_xy_px[:, 1],
            region.dorsal_boundary_xy_px[:, 0],
        )
    )
    palmar_x = float(
        np.interp(
            core_center_y,
            region.palmar_boundary_xy_px[:, 1],
            region.palmar_boundary_xy_px[:, 0],
        )
    )
    return dorsal_x / image_width, palmar_x / image_width


def _annotated_tile(rgb: np.ndarray, region, name: str) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    overlay[region.search_mask] = (255, 170, 0)
    bgr = cv2.addWeighted(bgr, 0.80, overlay, 0.20, 0.0)
    dorsal = np.rint(region.dorsal_boundary_xy_px).astype(np.int32)
    palmar = np.rint(region.palmar_boundary_xy_px).astype(np.int32)
    cv2.polylines(bgr, [dorsal], False, (255, 0, 255), 3, cv2.LINE_AA)
    cv2.polylines(bgr, [palmar], False, (0, 255, 255), 3, cv2.LINE_AA)
    target_width = 480
    target_height = round(target_width * rgb.shape[0] / rgb.shape[1])
    tile = cv2.resize(bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)
    cv2.putText(
        tile,
        name,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        name,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def _tile_row(tiles: list[np.ndarray]) -> np.ndarray:
    target_height = max(tile.shape[0] for tile in tiles)
    padded = []
    for tile in tiles:
        top = (target_height - tile.shape[0]) // 2
        bottom = target_height - tile.shape[0] - top
        padded.append(
            cv2.copyMakeBorder(
                tile,
                top,
                bottom,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(245, 245, 245),
            )
        )
    return np.hstack(padded)


def main() -> None:
    image_paths = [
        IMAGE_DIRECTORY / f"p{index}_Color.png" for index in range(7)
    ] + [IMAGE_DIRECTORY / f"p{index}d_Color.png" for index in range(6)]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing reference images: " + ", ".join(missing))

    # Exclude one-time OpenCV initialization from reported runtime.
    detect_fingertip_boundary(_load_rgb(image_paths[0]))
    normal_tiles: list[np.ndarray] = []
    dark_tiles: list[np.ndarray] = []
    normal_times_ms: list[float] = []
    high_resolution_times_ms: list[float] = []
    failures: list[str] = []

    for path in image_paths:
        rgb = _load_rgb(path)
        start = perf_counter()
        try:
            region = detect_fingertip_boundary(rgb)
        except RuntimeError as error:
            failures.append(f"{path.name}: detection failed: {error}")
            continue
        elapsed_ms = 1000.0 * (perf_counter() - start)
        times = high_resolution_times_ms if rgb.shape[0] > 720 else normal_times_ms
        times.append(elapsed_ms)

        measured = np.asarray(_center_boundaries(region, rgb.shape[1]))
        expected = np.asarray(EXPECTED_CENTER_BOUNDARIES[path.name])
        maximum_error = float(np.max(np.abs(measured - expected)))
        if maximum_error > MAXIMUM_CENTER_ERROR_WIDTH_FRACTION:
            failures.append(
                f"{path.name}: central boundary error {maximum_error:.4f} "
                f"exceeds {MAXIMUM_CENTER_ERROR_WIDTH_FRACTION:.4f} image widths"
            )
        print(
            f"{path.name}: PASS, {elapsed_ms:.1f} ms, "
            f"center=({measured[0]:.4f}, {measured[1]:.4f}), "
            f"width={region.estimated_pad_width_px:.1f} px"
        )
        tile = _annotated_tile(rgb, region, path.name)
        (dark_tiles if "d_" in path.name else normal_tiles).append(tile)

    if normal_tiles and dark_tiles:
        first_row = _tile_row(normal_tiles)
        second_row = _tile_row(dark_tiles)
        output_width = max(first_row.shape[1], second_row.shape[1])
        first_row = cv2.copyMakeBorder(
            first_row,
            0,
            0,
            0,
            output_width - first_row.shape[1],
            cv2.BORDER_CONSTANT,
            value=(245, 245, 245),
        )
        second_row = cv2.copyMakeBorder(
            second_row,
            0,
            0,
            0,
            output_width - second_row.shape[1],
            cv2.BORDER_CONSTANT,
            value=(245, 245, 245),
        )
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUTPUT_PATH), np.vstack((first_row, second_row)))

    for label, values in (
        ("640x480", normal_times_ms),
        ("1920x1080 production path", high_resolution_times_ms),
    ):
        print(
            f"{label} runtime: median={np.median(values):.1f} ms, "
            f"p95={np.percentile(values, 95):.1f} ms"
        )
    print(f"artifact: {OUTPUT_PATH}")
    if failures:
        raise RuntimeError("\n".join(failures))
    print("Result: PASS (13/13 reference boundaries)")


if __name__ == "__main__":
    main()

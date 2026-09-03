"""Validate smooth emissive-fingertip segmentation on reference images."""

from __future__ import annotations

import itertools
from pathlib import Path
import sys

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.localization.fingertip_segmentation import segment_fingertip  # noqa: E402


IMAGE_DIRECTORY = REPOSITORY_ROOT / "experiments" / "img"
OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "output"
    / "validation"
    / "emissive_fingertip_segmentation"
    / "segmentation_13_image_validation.png"
)


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"could not read reference image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] <= 0.0:
        raise RuntimeError("segmentation mask has no centroid")
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _pairwise_ious(masks: list[np.ndarray]) -> np.ndarray:
    values = []
    for first, second in itertools.combinations(masks, 2):
        union = np.count_nonzero(first | second)
        values.append(np.count_nonzero(first & second) / union)
    return np.asarray(values, dtype=np.float64)


def _report_stability(
    name: str,
    masks: list[np.ndarray],
    runtimes_ms: list[float],
) -> None:
    areas = np.asarray([np.count_nonzero(mask) for mask in masks], dtype=np.float64)
    centroids = np.asarray([_centroid(mask) for mask in masks])
    ious = _pairwise_ious(masks)
    print(f"{name} segmentation stability:")
    print(f"  area CV: {np.std(areas) / np.mean(areas):.6f}")
    print(f"  centroid x range: {np.ptp(centroids[:, 0]):.3f} px")
    print(f"  centroid y range: {np.ptp(centroids[:, 1]):.3f} px")
    print(f"  median pairwise mask IoU: {np.median(ious):.6f}")
    print(f"  minimum pairwise mask IoU: {np.min(ious):.6f}")
    print(f"  runtime median: {np.median(runtimes_ms):.3f} ms")
    print(f"  runtime p95: {np.percentile(runtimes_ms, 95):.3f} ms")


def _annotated_tile(rgb: np.ndarray, diagnostics, name: str) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    overlay[diagnostics.final_mask] = (80, 180, 40)
    bgr = cv2.addWeighted(bgr, 0.75, overlay, 0.25, 0.0)
    cv2.polylines(
        bgr,
        [np.rint(diagnostics.contour_xy_px).astype(np.int32)],
        True,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )
    fit_width = 400
    fit_height = 240
    scale = min(fit_width / bgr.shape[1], fit_height / bgr.shape[0])
    resized = cv2.resize(
        bgr,
        (round(scale * bgr.shape[1]), round(scale * bgr.shape[0])),
        interpolation=cv2.INTER_AREA,
    )
    tile = np.full((fit_height, fit_width, 3), 245, dtype=np.uint8)
    top = (fit_height - resized.shape[0]) // 2
    left = (fit_width - resized.shape[1]) // 2
    tile[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    cv2.putText(
        tile,
        name,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        name,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def main() -> None:
    sequences = {
        "Solaris p1-p6": [
            IMAGE_DIRECTORY / f"solaris_p{index}_Color.png"
            for index in range(1, 7)
        ],
        "Dragon Skin unloaded/p1/p3-p6": [
            IMAGE_DIRECTORY / "dragonskin_unloaded_Color.png",
            IMAGE_DIRECTORY / "dragonskin_p1_Color.png",
            *(
                IMAGE_DIRECTORY / f"dragonskin_p{index}_Color.png"
                for index in range(3, 7)
            ),
        ],
        "Dark room": [
            IMAGE_DIRECTORY / "solaris_unloaded_dark_Color.png"
        ],
    }
    paths = [path for sequence in sequences.values() for path in sequence]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing reference images: " + ", ".join(missing))

    segment_fingertip(_load_rgb(paths[0]))
    sequence_masks: dict[str, list[np.ndarray]] = {}
    sequence_runtimes: dict[str, list[float]] = {}
    tile_rows: list[np.ndarray] = []
    success_count = 0

    for sequence_name, sequence_paths in sequences.items():
        masks = []
        runtimes_ms = []
        tiles = []
        for path in sequence_paths:
            rgb = _load_rgb(path)
            diagnostics = segment_fingertip(rgb)
            masks.append(diagnostics.final_mask)
            runtimes_ms.append(diagnostics.runtime_ms)
            tiles.append(_annotated_tile(rgb, diagnostics, path.name))
            success_count += 1
            print(
                f"{path.name}: PASS, {diagnostics.runtime_ms:.1f} ms, "
                f"scale={diagnostics.geometry_scale:.4f}, "
                f"area={np.count_nonzero(diagnostics.final_mask)} px"
            )
        sequence_masks[sequence_name] = masks
        sequence_runtimes[sequence_name] = runtimes_ms
        tiles.extend([np.full_like(tiles[0], 245)] * (7 - len(tiles)))
        tile_rows.append(np.hstack(tiles))

    _report_stability(
        "Solaris p1-p6",
        sequence_masks["Solaris p1-p6"],
        sequence_runtimes["Solaris p1-p6"],
    )
    _report_stability(
        "Dragon Skin unloaded/p1/p3-p6",
        sequence_masks["Dragon Skin unloaded/p1/p3-p6"],
        sequence_runtimes["Dragon Skin unloaded/p1/p3-p6"],
    )
    dark_runtime = sequence_runtimes["Dark room"][0]
    print(f"Dark-room execution: PASS, runtime={dark_runtime:.3f} ms")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), np.vstack(tile_rows))
    print(f"artifact: {OUTPUT_PATH}")
    print(f"Result: PASS ({success_count}/13 fixed-parameter executions)")


if __name__ == "__main__":
    main()

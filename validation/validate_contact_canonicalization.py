"""Ablate only the longitudinal canonical span for Dragon Skin images."""

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
    CanonicalFingerConfig,
    CanonicalFingerMap,
    DenseProfileConfig,
    build_canonical_finger_map,
    extract_dense_response_profile,
    fit_affine_position_from_centroid,
    response_centroid,
    segment_fingertip,
    warp_to_canonical,
)


IMAGE_DIRECTORY = REPOSITORY_ROOT / "experiments" / "img"
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "output" / "validation" / "contact_canonicalization_ablation"
)
UNLOADED_FILE = "dragonskin_unloaded_Color.png"
LOADED_SEQUENCE = (
    ("dragonskin_p1_Color.png", 5.0),
    ("dragonskin_p3_Color.png", 10.0),
    ("dragonskin_p4_Color.png", 15.0),
    ("dragonskin_p5_Color.png", 20.0),
    ("dragonskin_p6_Color.png", 25.0),
)
TRANSVERSE_INSET_FRACTION = 0.04
HISTORICAL_MAE_MM = 1.12
PROFILE_CONFIG = DenseProfileConfig(
    mode="abs_highpass_red",
    transverse_start_fraction=0.0,
    transverse_stop_fraction=0.95,
    transverse_reduction="mean",
    longitudinal_smoothing_sigma_px=2.0,
    highpass_sigma_px=5.0,
)


def _load_rgb(filename: str) -> np.ndarray:
    path = IMAGE_DIRECTORY / filename
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read experiment image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _map_with_span(
    region,
    span: tuple[int, int],
    *,
    output_height: int,
    output_width: int,
) -> CanonicalFingerMap:
    """Build the current boundary interpolation with an explicit row span."""

    y_start, y_stop = span
    boundary_y = region.dorsal_boundary_xy_px[:, 1]
    if y_start < boundary_y[0] or y_stop - 1 > boundary_y[-1]:
        raise RuntimeError("requested canonical span exceeds segmented boundaries")
    source_y = np.linspace(y_start, y_stop - 1, output_height, dtype=np.float64)
    dorsal_x = np.interp(
        source_y,
        boundary_y,
        region.dorsal_boundary_xy_px[:, 0],
    )
    palmar_x = np.interp(
        source_y,
        boundary_y,
        region.palmar_boundary_xy_px[:, 0],
    )
    width = palmar_x - dorsal_x
    if np.any(width <= 0.0):
        raise RuntimeError("segmented dorsal and palmar boundaries cross")

    left = dorsal_x + TRANSVERSE_INSET_FRACTION * width
    right = palmar_x - TRANSVERSE_INSET_FRACTION * width
    transverse = np.linspace(0.0, 1.0, output_width, dtype=np.float64)
    map_x = left[:, None] + transverse[None, :] * (right - left)[:, None]
    map_y = np.broadcast_to(source_y[:, None], map_x.shape)
    return CanonicalFingerMap(map_x=map_x, map_y=map_y)


def _evaluate(
    images: list[np.ndarray],
    positions_mm: np.ndarray,
    canonical_map: CanonicalFingerMap,
) -> dict[str, np.ndarray | float | int]:
    canonical_images = [warp_to_canonical(image, canonical_map) for image in images]
    responses = np.vstack(
        [
            extract_dense_response_profile(
                image,
                canonical_images[0],
                PROFILE_CONFIG,
            )
            for image in canonical_images[1:]
        ]
    )
    centroids = np.asarray([response_centroid(response) for response in responses])

    full_model = fit_affine_position_from_centroid(centroids, positions_mm)
    fitted_predictions = (
        full_model.slope_mm * centroids + full_model.intercept_mm
    )
    loo_predictions = []
    for index in range(len(positions_mm)):
        keep = np.arange(len(positions_mm)) != index
        model = fit_affine_position_from_centroid(
            centroids[keep],
            positions_mm[keep],
        )
        loo_predictions.append(
            model.slope_mm * centroids[index] + model.intercept_mm
        )
    loo_predictions_array = np.asarray(loo_predictions)
    nearest_labels = positions_mm[
        np.argmin(
            np.abs(loo_predictions_array[:, None] - positions_mm[None, :]),
            axis=1,
        )
    ]
    return {
        "responses": responses,
        "centroids": centroids,
        "fitted_predictions_mm": fitted_predictions,
        "fitted_mae_mm": float(np.mean(np.abs(fitted_predictions - positions_mm))),
        "loo_predictions_mm": loo_predictions_array,
        "loo_mae_mm": float(
            np.mean(np.abs(loo_predictions_array - positions_mm))
        ),
        "nearest_position_count": int(
            np.count_nonzero(nearest_labels == positions_mm)
        ),
    }


def _write_results(
    results: dict[str, dict[str, object]],
    positions_mm: np.ndarray,
    full_span: tuple[int, int],
    core_span: tuple[int, int],
) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIRECTORY / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "variant",
                "source_y_start_px",
                "source_y_stop_px",
                "canonical_height_px",
                "canonical_width_px",
                "fitted_mae_mm",
                "loo_mae_mm",
                "nearest_position_count",
            )
        )
        for name, result in results.items():
            writer.writerow(
                (
                    name,
                    result["source_y_start_px"],
                    result["source_y_stop_px"],
                    result["canonical_height_px"],
                    result["canonical_width_px"],
                    result["fitted_mae_mm"],
                    result["loo_mae_mm"],
                    result["nearest_position_count"],
                )
            )

    with (OUTPUT_DIRECTORY / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("variant", "position_mm", "centroid", "loo_prediction_mm"))
        for name, result in results.items():
            for position, centroid, prediction in zip(
                positions_mm,
                result["centroids"],
                result["loo_predictions_mm"],
                strict=True,
            ):
                writer.writerow((name, position, centroid, prediction))

    lines = [
        "# Dragon Skin canonicalization ablation",
        "",
        "The segmentation result and optical descriptor are fixed. Only the",
        "longitudinal canonical span and, for `current_full`, the production",
        "canonical output resolution differ.",
        "",
        "## Fixed contract",
        "",
        f"- full active-row span: `{full_span}`",
        f"- segmentation core span: `{core_span}`",
        "- legacy/current-core canonical size: `250 x 120`",
        "- current-full canonical size: `256 x 128`",
        "- transverse inset: `0.04`",
        "- high-pass sigma: `5 px`",
        "- transverse reduction: mean over `0--95%`",
        "- longitudinal smoothing sigma: `2 px`",
        "- centroid: 10th-percentile floor then strongest 40% residual",
        f"- historical comparison value: `{HISTORICAL_MAE_MM:.2f} mm`",
        "",
        "## Results",
        "",
        "| Variant | Source rows | Shape | Fitted MAE [mm] | LOO MAE [mm] | Nearest labels |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['source_y_start_px']:.0f}--"
            f"{result['source_y_stop_px']:.0f} | "
            f"{result['canonical_height_px']} x {result['canonical_width_px']} | "
            f"{result['fitted_mae_mm']:.6f} | {result['loo_mae_mm']:.6f} | "
            f"{result['nearest_position_count']}/{len(positions_mm)} |"
        )
    legacy = results["legacy_exact"]
    core = results["current_core"]
    current_full = results["current_full"]
    lines.extend(
        (
            "",
            "## Direct comparisons",
            "",
            f"- legacy-exact minus historical LOO MAE: "
            f"`{legacy['loo_mae_mm'] - HISTORICAL_MAE_MM:+.6f} mm`",
            f"- current-core minus legacy-exact LOO MAE: "
            f"`{core['loo_mae_mm'] - legacy['loo_mae_mm']:+.6f} mm`",
            f"- current-full minus legacy-exact LOO MAE: "
            f"`{current_full['loo_mae_mm'] - legacy['loo_mae_mm']:+.6f} mm`",
            "",
            "No acceptance threshold or descriptor parameter was tuned for this ablation.",
        )
    )
    (OUTPUT_DIRECTORY / "report.md").write_text("\n".join(lines) + "\n")


def _write_figure(
    results: dict[str, dict[str, object]],
    positions_mm: np.ndarray,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.0), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(positions_mm)))
    for column, (name, result) in enumerate(results.items()):
        responses = result["responses"]
        for response, position, color in zip(
            responses,
            positions_mm,
            colors,
            strict=True,
        ):
            peak = float(np.max(response))
            normalized = response / peak if peak > 0.0 else response
            axes[0, column].plot(
                np.linspace(0.0, 1.0, len(response)),
                normalized,
                color=color,
                label=f"{position:g} mm",
            )
        axes[0, column].set(
            title=name.replace("_", " "),
            xlabel="Canonical longitudinal coordinate",
            ylabel="Peak-normalized response" if column == 0 else None,
            xlim=(0.0, 1.0),
        )
        axes[1, column].plot(
            positions_mm,
            positions_mm,
            color="0.65",
            linestyle="--",
        )
        axes[1, column].scatter(
            positions_mm,
            result["loo_predictions_mm"],
            color="#21918c",
            s=34,
        )
        axes[1, column].set(
            title=f"LOO MAE = {result['loo_mae_mm']:.2f} mm",
            xlabel="Contact position [mm]",
            ylabel="LOO estimate [mm]" if column == 0 else None,
        )
    axes[0, 0].legend(fontsize=8)
    figure.savefig(OUTPUT_DIRECTORY / "canonicalization_ablation.png", dpi=240)
    figure.savefig(OUTPUT_DIRECTORY / "canonicalization_ablation.pdf")
    plt.close(figure)


def main() -> None:
    filenames = [UNLOADED_FILE] + [filename for filename, _ in LOADED_SEQUENCE]
    positions_mm = np.asarray([position for _, position in LOADED_SEQUENCE])
    images = [_load_rgb(filename) for filename in filenames]

    cv2.setRNGSeed(0)
    segmentation = segment_fingertip(images[0])
    active_rows = np.flatnonzero(np.any(segmentation.final_mask, axis=1))
    if active_rows.size < 2:
        raise RuntimeError("segmentation final_mask has insufficient active rows")
    full_span = (int(active_rows[0]), int(active_rows[-1]) + 1)
    core_span = segmentation.region.core_y_span

    current_core = build_canonical_finger_map(
        segmentation.region,
        CanonicalFingerConfig(
            output_height=250,
            output_width=120,
            transverse_inset_fraction=TRANSVERSE_INSET_FRACTION,
            longitudinal_span="core",
        ),
    )
    rebuilt_core = _map_with_span(
        segmentation.region,
        core_span,
        output_height=250,
        output_width=120,
    )
    if not (
        np.array_equal(current_core.map_x, rebuilt_core.map_x)
        and np.array_equal(current_core.map_y, rebuilt_core.map_y)
    ):
        raise RuntimeError("explicit-span map does not reproduce production core map")

    current_full = build_canonical_finger_map(
        segmentation.region,
        CanonicalFingerConfig(
            transverse_inset_fraction=TRANSVERSE_INSET_FRACTION,
        ),
    )
    rebuilt_full = _map_with_span(
        segmentation.region,
        full_span,
        output_height=256,
        output_width=128,
    )
    if not (
        np.array_equal(current_full.map_x, rebuilt_full.map_x)
        and np.array_equal(current_full.map_y, rebuilt_full.map_y)
    ):
        raise RuntimeError("explicit-span map does not reproduce production full map")

    maps = {
        "legacy_exact": _map_with_span(
            segmentation.region,
            full_span,
            output_height=250,
            output_width=120,
        ),
        "current_core": current_core,
        "current_full": current_full,
    }

    results: dict[str, dict[str, object]] = {}
    for name, canonical_map in maps.items():
        result = _evaluate(images, positions_mm, canonical_map)
        result.update(
            {
                "source_y_start_px": float(canonical_map.map_y[0, 0]),
                "source_y_stop_px": float(canonical_map.map_y[-1, 0]),
                "canonical_height_px": canonical_map.output_height,
                "canonical_width_px": canonical_map.output_width,
            }
        )
        results[name] = result

    _write_results(results, positions_mm, full_span, core_span)
    _write_figure(results, positions_mm)

    print("Dragon Skin canonicalization ablation")
    print(f"  fixed final-mask full span: {full_span}")
    print(f"  segmentation core span:    {core_span}")
    for name, result in results.items():
        print(
            f"  {name:12s}: fitted MAE={result['fitted_mae_mm']:.6f} mm, "
            f"LOO MAE={result['loo_mae_mm']:.6f} mm, "
            f"nearest={result['nearest_position_count']}/{len(positions_mm)}"
        )
    print(f"Artifacts: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()

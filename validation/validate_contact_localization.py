"""Characterize shared optical contact observers on recorded RGB sequences."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.localization import (  # noqa: E402
    DenseProfileConfig,
    brightest_red_features,
    build_canonical_finger_map,
    build_dense_template_model,
    detect_fingertip_boundary,
    detect_led_array,
    estimate_contact_position,
    estimate_dense_template_position,
    extract_dense_profile,
    extract_dense_response_profile,
    fit_affine_position_from_centroid,
    mean_center_l2,
    response_centroid,
    save_dense_template_model,
    unloaded_baseline_statistics,
    warp_to_canonical,
)
IMAGE_DIRECTORY = REPOSITORY_ROOT / "experiments" / "img"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "output" / "validation" / "contact_localization"

# Experiment metadata is explicit rather than inferred from filename digits.
SOLARIS_SEQUENCE = (
    (("solaris_p1_Color.png", "p0_Color.png"), 0.0),
    (("solaris_p2_Color.png", "p1_Color.png"), 5.0),
    (("solaris_p3_Color.png", "p2_Color.png"), 10.0),
    (("p3_Color.png",), 15.0),
    (("solaris_p4_Color.png", "p4_Color.png"), 20.0),
    (("solaris_p5_Color.png", "p5_Color.png"), 25.0),
    (("solaris_p6_Color.png", "p6_Color.png"), 30.0),
)
DRAGON_UNLOADED_FILES = (
    "dragonskin_unloaded_Color.png",
    "p0d_Color.png",
)
DRAGON_LOADED_SEQUENCE = (
    (("dragonskin_p1_Color.png", "p1d_Color.png"), 5.0),
    (("dragonskin_p3_Color.png", "p2d_Color.png"), 10.0),
    (("dragonskin_p4_Color.png", "p3d_Color.png"), 15.0),
    (("dragonskin_p5_Color.png", "p4d_Color.png"), 20.0),
    (("dragonskin_p6_Color.png", "p5d_Color.png"), 25.0),
)
INDEPENDENT_VIEW_FILES = {
    "unloaded": "unloaded_Color.png",
    "distal_led1": "Loaded1_Color.png",
    "proximal_10mm": "Loaded2_Color.png",
}
INDEPENDENT_LED_POSITIONS_MM = np.asarray((-22, -11, 0, 11, 22), dtype=np.float64)
FEATURE_MODES = (
    "top10_red",
    "mean_red",
    "abs_highpass_red",
    "red_gradient",
    "red_fraction",
)


def _load_rgb(filename: str) -> np.ndarray:
    path = IMAGE_DIRECTORY / filename
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read experiment image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resolve_filename(candidates: tuple[str, ...]) -> str:
    for filename in candidates:
        if (IMAGE_DIRECTORY / filename).is_file():
            return filename
    raise FileNotFoundError(
        "missing explicitly named experiment image alternatives: "
        + ", ".join(candidates)
    )


def _canonical_sequence(filenames: list[str]) -> tuple[list[np.ndarray], tuple[int, int]]:
    images = [_load_rgb(filename) for filename in filenames]
    # GrabCut initialization uses OpenCV's process-global RNG. Reset it per
    # sequence so this offline characterization is reproducible and independent
    # of which material sequence runs first.
    cv2.setRNGSeed(0)
    reference_region = detect_fingertip_boundary(images[0])
    canonical_map = build_canonical_finger_map(reference_region)
    canonical_images = [warp_to_canonical(image, canonical_map) for image in images]
    return canonical_images, canonical_images[0].shape[:2]


def _normalized_profiles(
    canonical_images: list[np.ndarray],
    config: DenseProfileConfig,
) -> np.ndarray:
    return np.vstack(
        [
            mean_center_l2(extract_dense_profile(image, config))
            for image in canonical_images
        ]
    )


def _solaris_metrics(
    profiles: np.ndarray,
    positions_mm: np.ndarray,
) -> dict[str, object]:
    feature_distances = []
    physical_distances = []
    for first, second in itertools.combinations(range(len(profiles)), 2):
        feature_distances.append(float(np.linalg.norm(profiles[first] - profiles[second])))
        physical_distances.append(float(abs(positions_mm[first] - positions_mm[second])))
    feature_distances_array = np.asarray(feature_distances)
    physical_distances_array = np.asarray(physical_distances)
    distance_spearman = float(
        spearmanr(feature_distances_array, physical_distances_array).statistic
    )

    nearest_indices = []
    nearest_separations = []
    for index, profile in enumerate(profiles):
        distances = np.linalg.norm(profiles - profile, axis=1)
        distances[index] = np.inf
        nearest = int(np.argmin(distances))
        nearest_indices.append(nearest)
        nearest_separations.append(abs(positions_mm[index] - positions_mm[nearest]))

    centered = profiles - np.mean(profiles, axis=0)
    scores = np.linalg.svd(centered, full_matrices=False)[0][:, 0]
    pca_spearman = float(spearmanr(scores, positions_mm).statistic)
    if pca_spearman < 0.0:
        scores *= -1.0
        pca_spearman *= -1.0

    loo_predictions = []
    for index in range(len(profiles)):
        keep = np.arange(len(profiles)) != index
        model = build_dense_template_model(
            profiles[keep],
            positions_mm[keep],
            normalization="none",
        )
        estimate = estimate_dense_template_position(profiles[index], model)
        loo_predictions.append(estimate.position_mm)

    return {
        "feature_distances": feature_distances_array,
        "physical_distances": physical_distances_array,
        "distance_spearman": distance_spearman,
        "nearest_indices": np.asarray(nearest_indices, dtype=np.int64),
        "nearest_separations_mm": np.asarray(nearest_separations),
        "adjacent_nearest_count": int(np.count_nonzero(np.asarray(nearest_separations) == 5.0)),
        "pca_scores": scores,
        "pca_spearman": pca_spearman,
        "loo_predictions_mm": np.asarray(loo_predictions),
        "loo_mae_mm": float(np.mean(np.abs(np.asarray(loo_predictions) - positions_mm))),
    }


def _dragon_metrics(
    responses: np.ndarray,
    positions_mm: np.ndarray,
) -> dict[str, object]:
    centroids = np.asarray([response_centroid(response) for response in responses])
    predictions = []
    for index in range(len(positions_mm)):
        keep = np.arange(len(positions_mm)) != index
        model = fit_affine_position_from_centroid(
            centroids[keep],
            positions_mm[keep],
        )
        predictions.append(model.slope_mm * centroids[index] + model.intercept_mm)
    predictions = np.asarray(predictions)
    nearest_labels = positions_mm[
        np.argmin(np.abs(predictions[:, None] - positions_mm[None, :]), axis=1)
    ]
    pairwise_distances = np.asarray(
        [
            np.linalg.norm(first - second)
            for first, second in itertools.combinations(responses, 2)
        ]
    )
    return {
        "responses": responses,
        "centroids": centroids,
        "loo_predictions_mm": predictions,
        "loo_mae_mm": float(np.mean(np.abs(predictions - positions_mm))),
        "nearest_position_count": int(np.count_nonzero(nearest_labels == positions_mm)),
        "minimum_pairwise_response_distance": float(np.min(pairwise_distances)),
    }


def _independent_led_sanity() -> str:
    paths = {name: IMAGE_DIRECTORY / filename for name, filename in INDEPENDENT_VIEW_FILES.items()}
    if not all(path.is_file() for path in paths.values()):
        missing = ", ".join(path.name for path in paths.values() if not path.is_file())
        return f"SKIP (missing explicitly named files: {missing})"
    frames = {name: _load_rgb(path.name) for name, path in paths.items()}
    geometry = detect_led_array(
        np.stack((frames["unloaded"], frames["distal_led1"], frames["proximal_10mm"]))
    )
    baseline_features = brightest_red_features(frames["unloaded"], geometry)
    baseline, noise = unloaded_baseline_statistics(np.repeat(baseline_features[None], 3, axis=0))
    estimates = [
        estimate_contact_position(
            brightest_red_features(frames[name], geometry),
            baseline,
            noise,
            INDEPENDENT_LED_POSITIONS_MM,
        )
        for name in ("distal_led1", "proximal_10mm")
    ]
    if not all(estimate.contact_detected for estimate in estimates):
        return "FAIL (one or both loaded frames did not cross the contact gate)"
    if estimates[0].predicted_led_index == estimates[1].predicted_led_index:
        return "FAIL (Loaded1 and Loaded2 selected the same peak LED)"
    return "PASS (loaded frames select distinct peak LEDs)"


def _write_figure(
    solaris_profiles: np.ndarray,
    solaris_positions: np.ndarray,
    solaris_metrics: dict[str, object],
    dragon_metrics: dict[str, object],
    dragon_positions: np.ndarray,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 6.5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(solaris_profiles)))
    for profile, position, color in zip(
        solaris_profiles, solaris_positions, colors, strict=True
    ):
        axes[0, 0].plot(profile, color=color, label=f"{position:g} mm")
    axes[0, 0].set(title="Solaris dense top-10% red", xlabel="Canonical longitudinal sample", ylabel="Normalized response")
    axes[0, 0].legend(ncol=2, fontsize=8)

    axes[0, 1].scatter(
        solaris_metrics["physical_distances"],
        solaris_metrics["feature_distances"],
        color="#355f8d",
        s=26,
    )
    axes[0, 1].set(
        title=f"Pairwise ordering (Spearman={solaris_metrics['distance_spearman']:.3f})",
        xlabel="Physical separation [mm]",
        ylabel="Feature distance",
    )

    axes[1, 0].plot(
        solaris_positions,
        solaris_metrics["pca_scores"],
        "o-",
        color="#21918c",
    )
    axes[1, 0].set(
        title=f"Descriptive PCA-1 ordering (Spearman={solaris_metrics['pca_spearman']:.3f})",
        xlabel="Contact position [mm]",
        ylabel="PCA-1 score",
    )

    axes[1, 1].plot(dragon_positions, dragon_positions, color="0.65", linestyle="--")
    axes[1, 1].scatter(
        dragon_positions,
        dragon_metrics["loo_predictions_mm"],
        color="#5ec962",
        s=38,
    )
    axes[1, 1].set(
        title=f"Dragon unloaded-relative centroid (LOO MAE={dragon_metrics['loo_mae_mm']:.2f} mm)",
        xlabel="Contact position [mm]",
        ylabel="Leave-one-out estimate [mm]",
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIRECTORY / "contact_localization.png", dpi=200)
    figure.savefig(OUTPUT_DIRECTORY / "contact_localization.pdf")
    plt.close(figure)


def _write_csv(
    solaris_positions: np.ndarray,
    solaris_metrics: dict[str, object],
    dragon_positions: np.ndarray,
    dragon_metrics: dict[str, object],
) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIRECTORY / "position_estimates.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("material", "position_mm", "estimate_mm", "nearest_other_separation_mm"))
        for index, position in enumerate(solaris_positions):
            writer.writerow(
                (
                    "Solaris",
                    position,
                    solaris_metrics["loo_predictions_mm"][index],
                    solaris_metrics["nearest_separations_mm"][index],
                )
            )
        for index, position in enumerate(dragon_positions):
            writer.writerow(
                (
                    "Dragon Skin",
                    position,
                    dragon_metrics["loo_predictions_mm"][index],
                    "",
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-features",
        action="store_true",
        help="compare the five fixed dense feature definitions",
    )
    parser.add_argument(
        "--export-template",
        type=Path,
        help="write the Solaris dense-top10 templates to this NPZ path",
    )
    arguments = parser.parse_args()

    solaris_filenames = [
        _resolve_filename(filenames) for filenames, _ in SOLARIS_SEQUENCE
    ]
    solaris_positions = np.asarray([position for _, position in SOLARIS_SEQUENCE])
    solaris_images, canonical_shape = _canonical_sequence(solaris_filenames)
    solaris_config = DenseProfileConfig(mode="top10_red")
    solaris_raw_profiles = np.vstack(
        [extract_dense_profile(image, solaris_config) for image in solaris_images]
    )
    solaris_profiles = np.vstack(
        [mean_center_l2(profile) for profile in solaris_raw_profiles]
    )
    solaris_result = _solaris_metrics(solaris_profiles, solaris_positions)

    dragon_filenames = [_resolve_filename(DRAGON_UNLOADED_FILES)] + [
        _resolve_filename(filenames) for filenames, _ in DRAGON_LOADED_SEQUENCE
    ]
    dragon_positions = np.asarray([position for _, position in DRAGON_LOADED_SEQUENCE])
    dragon_images, _ = _canonical_sequence(dragon_filenames)
    dragon_config = DenseProfileConfig(mode="abs_highpass_red")
    dragon_profiles = np.vstack(
        [
            extract_dense_response_profile(
                image,
                dragon_images[0],
                dragon_config,
            )
            for image in dragon_images[1:]
        ]
    )
    dragon_result = _dragon_metrics(dragon_profiles, dragon_positions)

    print("Solaris dense top-10% red:")
    print(f"  pairwise distance/physical-distance Spearman: {solaris_result['distance_spearman']:.6f}")
    print(
        "  nearest other frame is adjacent 5 mm: "
        f"{solaris_result['adjacent_nearest_count']}/{len(solaris_positions)}"
    )
    print(f"  PCA-1 ordering Spearman: {solaris_result['pca_spearman']:.6f}")
    print(f"  leave-one-position-out template MAE: {solaris_result['loo_mae_mm']:.3f} mm")
    print("Dragon Skin unloaded-relative dense high-pass response:")
    print(f"  leave-one-position-out centroid MAE: {dragon_result['loo_mae_mm']:.3f} mm")
    print(
        "  nearest labelled position: "
        f"{dragon_result['nearest_position_count']}/{len(dragon_positions)}"
    )
    print(
        "  minimum pairwise response distance: "
        f"{dragon_result['minimum_pairwise_response_distance']:.6f}"
    )
    independent_result = _independent_led_sanity()
    print(f"Independent Loaded1/Loaded2 LED-patch sanity: {independent_result}")

    if arguments.compare_features:
        print("Optional Solaris feature comparison:")
        for mode in FEATURE_MODES:
            profiles = _normalized_profiles(
                solaris_images,
                DenseProfileConfig(mode=mode),
            )
            metrics = _solaris_metrics(profiles, solaris_positions)
            print(
                f"  {mode}: Spearman={metrics['distance_spearman']:.4f}, "
                f"adjacent={metrics['adjacent_nearest_count']}/{len(solaris_positions)}"
            )

    if arguments.export_template is not None:
        model = build_dense_template_model(
            solaris_raw_profiles,
            solaris_positions,
            canonical_shape=canonical_shape,
            feature_config=solaris_config,
            normalization="mean_center_l2",
        )
        save_dense_template_model(arguments.export_template, model)
        print(f"Dense template model: {arguments.export_template}")

    _write_csv(solaris_positions, solaris_result, dragon_positions, dragon_result)
    _write_figure(
        solaris_profiles,
        solaris_positions,
        solaris_result,
        dragon_result,
        dragon_positions,
    )
    print(f"Artifacts: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()

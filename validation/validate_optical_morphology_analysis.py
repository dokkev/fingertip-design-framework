"""Characterize fixed-frame Dragon Skin optical magnitude and separability."""

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

from experiments.localization import calibrate_fixed_finger  # noqa: E402
from experiments.optical_morphology_analysis import (  # noqa: E402
    fixed_red_differences,
    longitudinal_red_signatures,
    mean_absolute_red_response,
    minimum_pairwise_separation,
    pairwise_signature_distances,
)


IMAGE_DIRECTORY = REPOSITORY_ROOT / "experiments" / "img"
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "output" / "validation" / "optical_morphology_analysis"
)
UNLOADED_FILENAME = "dragonskin_unloaded_Color.png"
LOADED_SEQUENCE = (
    ("dragonskin_p1_Color.png", 5.0),
    ("dragonskin_p3_Color.png", 10.0),
    ("dragonskin_p4_Color.png", 15.0),
    ("dragonskin_p5_Color.png", 20.0),
    ("dragonskin_p6_Color.png", 25.0),
)


def _load_rgb(filename: str) -> np.ndarray:
    bgr = cv2.imread(str(IMAGE_DIRECTORY / filename), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"missing experiment image: {filename}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    unloaded = _load_rgb(UNLOADED_FILENAME)
    loaded = np.stack([_load_rgb(filename) for filename, _ in LOADED_SEQUENCE])
    positions_mm = np.asarray([position for _, position in LOADED_SEQUENCE])
    cv2.setRNGSeed(0)
    calibration = calibrate_fixed_finger(unloaded)
    differences = fixed_red_differences(unloaded, loaded, calibration)
    magnitudes = mean_absolute_red_response(differences)
    signatures = longitudinal_red_signatures(differences)
    distances = pairwise_signature_distances(signatures)
    minimum, pair = minimum_pairwise_separation(distances)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIRECTORY / "response_magnitude.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("filename", "contact_position_mm", "mean_absolute_delta_red_dn"))
        for (filename, position), magnitude in zip(
            LOADED_SEQUENCE,
            magnitudes,
            strict=True,
        ):
            writer.writerow((filename, position, magnitude))
    np.savetxt(
        OUTPUT_DIRECTORY / "pairwise_signature_distance_dn.csv",
        distances,
        delimiter=",",
        header=",".join(f"{position:g}mm" for position in positions_mm),
        comments="",
    )

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), constrained_layout=True)
    axes[0].bar(positions_mm, magnitudes, width=3.5, color="#31688e")
    axes[0].set(
        title="Optical-response magnitude",
        xlabel="Contact position [mm]",
        ylabel=r"Mean $|\Delta R|$ [DN]",
    )
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(signatures)))
    longitudinal = np.linspace(0.0, 1.0, signatures.shape[1])
    for signature, position, color in zip(
        signatures,
        positions_mm,
        colors,
        strict=True,
    ):
        axes[1].plot(longitudinal, signature, color=color, label=f"{position:g} mm")
    axes[1].set(
        title="Longitudinal signatures",
        xlabel="Fixed longitudinal coordinate",
        ylabel=r"Mean $\Delta R$ [DN]",
    )
    axes[1].legend(fontsize=8, ncol=2)
    image = axes[2].imshow(distances, cmap="viridis")
    axes[2].set(
        title="Pairwise signature RMS",
        xlabel="Contact state",
        ylabel="Contact state",
        xticks=np.arange(len(positions_mm)),
        yticks=np.arange(len(positions_mm)),
        xticklabels=[f"{value:g}" for value in positions_mm],
        yticklabels=[f"{value:g}" for value in positions_mm],
    )
    figure.colorbar(image, ax=axes[2], label="RMS distance [DN]")
    figure.savefig(OUTPUT_DIRECTORY / "optical_morphology_analysis.png", dpi=200)
    figure.savefig(OUTPUT_DIRECTORY / "optical_morphology_analysis.pdf")
    plt.close(figure)

    print("Dragon Skin fixed-frame optical morphology analysis")
    print(f"  response magnitude [DN]: {np.round(magnitudes, 6).tolist()}")
    print(f"  minimum signature separation: {minimum:.6f} DN")
    print(
        "  limiting positions: "
        f"{positions_mm[pair[0]]:g} vs {positions_mm[pair[1]]:g} mm"
    )
    print("  Solaris: not evaluated (no same-condition unloaded reference image)")
    print(f"Artifacts: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()

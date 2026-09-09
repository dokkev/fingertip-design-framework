"""Simple contact-location decoding for paper Figures 5(c) and 6."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import yaml

from algorithm.contact_localization import (
    ContactLocalizationConfig,
    UnloadedOpticalReference,
    build_unloaded_reference,
    compute_longitudinal_response,
    localize_response_profile,
)
from lumo.fingertip.layout import LED_CENTERS_Y_MM, TOTAL_Y_BOUNDS_MM
from lumo.visualization import MATERIAL_LABELS, PAPER_COLORS, PAPER_LABELS


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "paper_figures.yaml"
DEFAULT_CONTACT_DATASET_H5 = REPOSITORY_ROOT / "output" / "upload" / "contact_dataset.h5"


@dataclass(frozen=True)
class PaperFigureConfig:
    """Explicit data, display, and evaluation contract for Figures 5(c) and 6."""

    analysis_output_directory: Path
    figure5_output_directory: Path
    figure6_output_directory: Path
    profile_roots: dict[str, Path]
    condition_overrides: dict[tuple[str, str, str], Path]
    materials: tuple[str, ...]
    indenters: tuple[str, ...]
    morphologies: tuple[str, ...]
    material_labels: dict[str, str]
    indenter_labels: dict[str, str]
    morphology_labels: dict[str, str]
    morphology_colors: dict[str, str]
    contact_positions_mm: dict[int, float]
    low_force_n: float
    high_force_n: float
    region_count: int
    maximum_calibration_contacts: int
    maximum_resamples: int
    random_seed: int

    def source_root(self, material: str, morphology: str, indenter: str) -> Path:
        """Return the profile/metric root for one measured condition."""

        return self.condition_overrides.get(
            (material, morphology, indenter), self.profile_roots[material]
        )

    def condition_label(self, material: str, indenter: str) -> str:
        """Return the paper-facing material/indenter label."""

        return f"{self.material_labels[material]} · {self.indenter_labels[indenter]}"


@dataclass(frozen=True)
class DecoderSample:
    """One independent repeated contact represented by six optical regions."""

    material: str
    indenter: str
    morphology: str
    specimen_id: str
    run_id: str
    hole_index: int
    repetition_index: int
    actual_force_low_n: float
    actual_force_high_n: float
    feature: np.ndarray
    source_path: Path

    @property
    def scalar_magnitude(self) -> float:
        """Return the scalar-only optical representation used in Figure 6."""

        return float(np.linalg.norm(self.feature))


def _resolved_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> PaperFigureConfig:
    """Load and validate the paper-figure YAML configuration."""

    config_path = _resolved_path(path)
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("paper figure config must contain a mapping")

    labels = raw["display_labels"]
    materials = tuple(str(value) for value in raw["materials"])
    indenters = tuple(str(value) for value in raw["indenters"])
    morphologies = tuple(str(value) for value in raw["morphologies"])
    profile_roots = {
        str(material): _resolved_path(root)
        for material, root in raw["profile_roots"].items()
    }
    overrides = {
        (
            str(entry["material"]),
            str(entry["morphology"]),
            str(entry["indenter"]),
        ): _resolved_path(entry["root"])
        for entry in raw.get("condition_overrides", ())
    }
    positions = {
        int(hole): float(position)
        for hole, position in raw["contact_positions_mm"].items()
    }
    feature = raw["feature"]
    calibration = raw["calibration"]
    config = PaperFigureConfig(
        analysis_output_directory=_resolved_path(raw["analysis_output_directory"]),
        figure5_output_directory=_resolved_path(raw["figure5_output_directory"]),
        figure6_output_directory=_resolved_path(raw["figure6_output_directory"]),
        profile_roots=profile_roots,
        condition_overrides=overrides,
        materials=materials,
        indenters=indenters,
        morphologies=morphologies,
        material_labels={material: MATERIAL_LABELS[material] for material in materials},
        indenter_labels={
            str(key): str(value) for key, value in labels["indenters"].items()
        },
        morphology_labels={
            morphology: PAPER_LABELS[morphology] for morphology in morphologies
        },
        morphology_colors={
            morphology: PAPER_COLORS[morphology] for morphology in morphologies
        },
        contact_positions_mm=positions,
        low_force_n=float(feature["low_force_n"]),
        high_force_n=float(feature["high_force_n"]),
        region_count=int(feature["region_count"]),
        maximum_calibration_contacts=int(calibration["maximum_contacts_per_location"]),
        maximum_resamples=int(calibration["maximum_resamples"]),
        random_seed=int(calibration["random_seed"]),
    )
    _validate_config(config)
    return config


def _validate_config(config: PaperFigureConfig) -> None:
    if config.high_force_n <= config.low_force_n:
        raise ValueError("high feature force must exceed low feature force")
    if config.region_count != len(config.contact_positions_mm):
        raise ValueError("region count must match the number of contact locations")
    if sorted(config.contact_positions_mm) != list(
        range(1, len(config.contact_positions_mm) + 1)
    ):
        raise ValueError("contact hole indices must be consecutive and one-based")
    if config.maximum_calibration_contacts < 1 or config.maximum_resamples < 1:
        raise ValueError("calibration limits must be positive")
    for material in config.materials:
        if (
            material not in config.profile_roots
            or material not in config.material_labels
        ):
            raise ValueError(
                f"missing source or display label for material {material!r}"
            )
    for indenter in config.indenters:
        if indenter not in config.indenter_labels:
            raise ValueError(f"missing display label for indenter {indenter!r}")
    for morphology in config.morphologies:
        if (
            morphology not in config.morphology_labels
            or morphology not in config.morphology_colors
        ):
            raise ValueError(f"missing label or color for morphology {morphology!r}")


def _region_indices(
    coordinate: np.ndarray, region_count: int
) -> tuple[np.ndarray, ...]:
    coordinate = np.asarray(coordinate, dtype=np.float64)
    if (
        coordinate.ndim != 1
        or len(coordinate) < region_count
        or not np.all(np.isfinite(coordinate))
        or not np.all(np.diff(coordinate) > 0.0)
        or not np.isclose(coordinate[0], 0.0)
        or not np.isclose(coordinate[-1], 1.0)
    ):
        raise ValueError(
            "profile coordinate must be finite, increasing, and span 0 to 1"
        )
    edges = np.linspace(0.0, 1.0, region_count + 1)
    output = []
    for region in range(region_count):
        upper = (
            coordinate <= edges[region + 1]
            if region == region_count - 1
            else coordinate < edges[region + 1]
        )
        indices = np.flatnonzero((coordinate >= edges[region]) & upper)
        if indices.size == 0:
            raise ValueError(f"optical region {region + 1} contains no profile bins")
        output.append(indices)
    return tuple(output)


def regionize_profile_change(
    low_profile: np.ndarray,
    high_profile: np.ndarray,
    coordinate: np.ndarray,
    *,
    region_count: int = 6,
) -> np.ndarray:
    """Return signed regional means of the high-minus-low optical profile."""

    low = np.asarray(low_profile, dtype=np.float64)
    high = np.asarray(high_profile, dtype=np.float64)
    if low.ndim != 1 or high.shape != low.shape or len(low) != len(coordinate):
        raise ValueError("low/high profiles and coordinate must have matching lengths")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("optical profiles must be finite")
    delta = high - low
    return np.asarray(
        [
            float(np.mean(delta[indices]))
            for indices in _region_indices(coordinate, region_count)
        ],
        dtype=np.float64,
    )


def load_condition_samples(
    config: PaperFigureConfig,
    material: str,
    indenter: str,
    morphology: str,
) -> tuple[list[DecoderSample], str]:
    """Load one condition from the compact profile artifact."""

    root = config.source_root(material, morphology, indenter)
    path = root / "raw_data_summary" / "longitudinal_profiles.npz"
    if not path.is_file():
        return [], f"missing profile artifact: {path}"

    with np.load(path, allow_pickle=False) as data:
        coordinate = np.asarray(data["longitudinal_coordinate"], dtype=np.float64)
        profiles = np.asarray(data["profiles"], dtype=np.float64)
        metadata = {
            name: np.asarray(data[name])
            for name in (
                "specimen_id",
                "material",
                "morphology",
                "run_id",
                "run_status",
                "indenter",
                "hole_index",
                "repetition_index",
                "target_force_n",
                "actual_force_n",
            )
        }
    if profiles.ndim != 2 or profiles.shape[1] != len(coordinate):
        raise ValueError(f"invalid profile matrix in {path}")

    mask = (
        (metadata["material"].astype(str) == material)
        & (metadata["morphology"].astype(str) == morphology)
        & (metadata["indenter"].astype(str) == indenter)
        & (metadata["run_status"].astype(str) == "complete")
    )
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return [], f"condition absent from {path}"

    groups: dict[tuple[str, str, int, int], list[int]] = {}
    for index in selected:
        key = (
            str(metadata["specimen_id"][index]),
            str(metadata["run_id"][index]),
            int(metadata["hole_index"][index]),
            int(metadata["repetition_index"][index]),
        )
        groups.setdefault(key, []).append(int(index))

    samples: list[DecoderSample] = []
    for (specimen_id, run_id, hole, repetition), indices in sorted(groups.items()):
        target = np.asarray(metadata["target_force_n"][indices], dtype=np.float64)
        low_indices = [
            indices[position]
            for position in np.flatnonzero(np.isclose(target, config.low_force_n))
        ]
        high_indices = [
            indices[position]
            for position in np.flatnonzero(np.isclose(target, config.high_force_n))
        ]
        if len(low_indices) != 1 or len(high_indices) != 1:
            raise ValueError(
                f"{path}: {run_id} requires exactly one {config.low_force_n:g} N "
                f"and one {config.high_force_n:g} N profile"
            )
        low_index = low_indices[0]
        high_index = high_indices[0]
        samples.append(
            DecoderSample(
                material=material,
                indenter=indenter,
                morphology=morphology,
                specimen_id=specimen_id,
                run_id=run_id,
                hole_index=hole,
                repetition_index=repetition,
                actual_force_low_n=float(metadata["actual_force_n"][low_index]),
                actual_force_high_n=float(metadata["actual_force_n"][high_index]),
                feature=regionize_profile_change(
                    profiles[low_index],
                    profiles[high_index],
                    coordinate,
                    region_count=config.region_count,
                ),
                source_path=path,
            )
        )

    expected_holes = set(config.contact_positions_mm)
    observed_holes = {sample.hole_index for sample in samples}
    repetitions_by_hole = {
        hole: {
            sample.repetition_index for sample in samples if sample.hole_index == hole
        }
        for hole in expected_holes
    }
    if observed_holes != expected_holes:
        return (
            [],
            f"expected holes {sorted(expected_holes)}, found {sorted(observed_holes)}",
        )
    if any(len(repetitions) < 2 for repetitions in repetitions_by_hole.values()):
        return [], "each contact location requires at least two independent repetitions"
    return samples, "measured"


def _predict(
    feature: np.ndarray,
    templates: dict[int, np.ndarray],
) -> tuple[int, float, float]:
    distances = sorted(
        (float(np.linalg.norm(feature - template)), hole)
        for hole, template in templates.items()
    )
    margin = distances[1][0] - distances[0][0] if len(distances) > 1 else float("nan")
    return distances[0][1], distances[0][0], margin


def decode_leave_one_repetition_out(
    samples: list[DecoderSample],
    config: PaperFigureConfig,
) -> list[dict[str, Any]]:
    """Decode each run using templates that exclude its repetition index."""

    output: list[dict[str, Any]] = []
    holes = tuple(config.contact_positions_mm)
    for sample in samples:
        spatial_templates: dict[int, np.ndarray] = {}
        scalar_templates: dict[int, np.ndarray] = {}
        for hole in holes:
            training = [
                candidate
                for candidate in samples
                if candidate.hole_index == hole
                and candidate.repetition_index != sample.repetition_index
            ]
            if not training:
                raise ValueError(
                    f"no training data for hole {hole} after holding out repetition "
                    f"{sample.repetition_index}"
                )
            spatial_templates[hole] = np.mean(
                np.asarray([candidate.feature for candidate in training]), axis=0
            )
            scalar_templates[hole] = np.asarray(
                [np.mean([candidate.scalar_magnitude for candidate in training])],
                dtype=np.float64,
            )

        spatial_prediction, spatial_distance, spatial_margin = _predict(
            sample.feature, spatial_templates
        )
        scalar_prediction, scalar_distance, scalar_margin = _predict(
            np.asarray([sample.scalar_magnitude]), scalar_templates
        )
        row: dict[str, Any] = {
            "material": sample.material,
            "material_display": config.material_labels[sample.material],
            "indenter": sample.indenter,
            "indenter_display": config.indenter_labels[sample.indenter],
            "morphology_id": sample.morphology,
            "morphology_display": config.morphology_labels[sample.morphology],
            "specimen_id": sample.specimen_id,
            "run_id": sample.run_id,
            "repetition_index": sample.repetition_index,
            "true_hole_index": sample.hole_index,
            "true_contact_position_mm": config.contact_positions_mm[sample.hole_index],
            "predicted_hole_spatial": spatial_prediction,
            "predicted_contact_position_spatial_mm": config.contact_positions_mm[
                spatial_prediction
            ],
            "spatial_correct": spatial_prediction == sample.hole_index,
            "spatial_neighbor_tolerant_correct": abs(
                spatial_prediction - sample.hole_index
            )
            <= 1,
            "spatial_nearest_distance": spatial_distance,
            "spatial_class_margin": spatial_margin,
            "predicted_hole_scalar": scalar_prediction,
            "predicted_contact_position_scalar_mm": config.contact_positions_mm[
                scalar_prediction
            ],
            "scalar_correct": scalar_prediction == sample.hole_index,
            "scalar_nearest_distance": scalar_distance,
            "scalar_class_margin": scalar_margin,
            "scalar_magnitude": sample.scalar_magnitude,
            "actual_force_low_n": sample.actual_force_low_n,
            "actual_force_high_n": sample.actual_force_high_n,
            "source_path": _display_path(sample.source_path),
        }
        for region, value in enumerate(sample.feature, start=1):
            row[f"region_{region}_delta_dn"] = float(value)
        output.append(row)
    return output


def _percent(values: list[bool]) -> float:
    return 100.0 * float(np.mean(np.asarray(values, dtype=np.float64)))


def _iqr(values: list[float]) -> float:
    return float(np.percentile(values, 75.0) - np.percentile(values, 25.0))


def _missing_summary_row(
    config: PaperFigureConfig,
    material: str,
    indenter: str,
    morphology: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "material": material,
        "material_display": config.material_labels[material],
        "indenter": indenter,
        "indenter_display": config.indenter_labels[indenter],
        "morphology_id": morphology,
        "morphology_display": config.morphology_labels[morphology],
        "feature_force_low_n": config.low_force_n,
        "feature_force_high_n": config.high_force_n,
        "feature_region_count": config.region_count,
        "n_test_samples": 0,
        "n_correct": "",
        "decoding_error_count": "",
        "accuracy_exact": "",
        "accuracy_neighbor_tolerant": "",
        "accuracy_scalar_only": "",
        "median_class_margin": "",
        "optical_magnitude_median_dn": "",
        "optical_magnitude_iqr_dn": "",
        "baseline_accuracy_for_condition": "",
        "improvement_abs": "",
        "improvement_rel": "",
        "status": f"unavailable: {reason}",
    }


def _summarize_predictions(
    config: PaperFigureConfig,
    material: str,
    indenter: str,
    morphology: str,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    magnitudes = [float(row["scalar_magnitude"]) for row in predictions]
    correct_count = sum(bool(row["spatial_correct"]) for row in predictions)
    return {
        "material": material,
        "material_display": config.material_labels[material],
        "indenter": indenter,
        "indenter_display": config.indenter_labels[indenter],
        "morphology_id": morphology,
        "morphology_display": config.morphology_labels[morphology],
        "feature_force_low_n": config.low_force_n,
        "feature_force_high_n": config.high_force_n,
        "feature_region_count": config.region_count,
        "n_test_samples": len(predictions),
        "n_correct": correct_count,
        "decoding_error_count": len(predictions) - correct_count,
        "accuracy_exact": 100.0 * correct_count / len(predictions),
        "accuracy_neighbor_tolerant": _percent(
            [bool(row["spatial_neighbor_tolerant_correct"]) for row in predictions]
        ),
        "accuracy_scalar_only": _percent(
            [bool(row["scalar_correct"]) for row in predictions]
        ),
        "median_class_margin": float(
            np.median([float(row["spatial_class_margin"]) for row in predictions])
        ),
        "optical_magnitude_median_dn": float(np.median(magnitudes)),
        "optical_magnitude_iqr_dn": _iqr(magnitudes),
        "baseline_accuracy_for_condition": "",
        "improvement_abs": "",
        "improvement_rel": "",
        "status": "measured",
    }


def _attach_baseline_comparisons(
    rows: list[dict[str, Any]], config: PaperFigureConfig
) -> None:
    lookup = {
        (str(row["material"]), str(row["indenter"]), str(row["morphology_id"])): row
        for row in rows
    }
    for material in config.materials:
        for indenter in config.indenters:
            baseline = lookup[(material, indenter, "baseline")]
            if baseline["status"] != "measured":
                continue
            baseline_accuracy = float(baseline["accuracy_exact"])
            for morphology in config.morphologies:
                row = lookup[(material, indenter, morphology)]
                if row["status"] != "measured":
                    continue
                accuracy = float(row["accuracy_exact"])
                row["baseline_accuracy_for_condition"] = baseline_accuracy
                row["improvement_abs"] = accuracy - baseline_accuracy
                row["improvement_rel"] = (
                    100.0 * (accuracy / baseline_accuracy - 1.0)
                    if baseline_accuracy > 0.0
                    else float("nan")
                )


def _baseline_relative_change_rows(
    summaries: list[dict[str, Any]], config: PaperFigureConfig
) -> list[dict[str, Any]]:
    """Return the eight optimized-versus-baseline comparisons for Figure 6(a)."""

    lookup = {
        (str(row["material"]), str(row["indenter"]), str(row["morphology_id"])): row
        for row in summaries
    }
    output: list[dict[str, Any]] = []
    for material in config.materials:
        for indenter in config.indenters:
            baseline = lookup[(material, indenter, "baseline")]
            for morphology in config.morphologies:
                if morphology == "baseline":
                    continue
                optimized = lookup[(material, indenter, morphology)]
                measured = (
                    baseline["status"] == "measured"
                    and optimized["status"] == "measured"
                )
                if measured:
                    baseline_magnitude = float(baseline["optical_magnitude_median_dn"])
                    optimized_magnitude = float(
                        optimized["optical_magnitude_median_dn"]
                    )
                    if baseline_magnitude <= 0.0:
                        raise ValueError(
                            f"non-positive baseline optical magnitude for "
                            f"{material}/{indenter}"
                        )
                    baseline_accuracy = float(baseline["accuracy_exact"])
                    optimized_accuracy = float(optimized["accuracy_exact"])
                    magnitude_change = 100.0 * (
                        optimized_magnitude / baseline_magnitude - 1.0
                    )
                    accuracy_change = optimized_accuracy - baseline_accuracy
                else:
                    baseline_magnitude = ""
                    optimized_magnitude = ""
                    magnitude_change = ""
                    baseline_accuracy = ""
                    optimized_accuracy = ""
                    accuracy_change = ""
                output.append(
                    {
                        "material": material,
                        "indenter": indenter,
                        "morphology": morphology,
                        "material_display": config.material_labels[material],
                        "indenter_display": config.indenter_labels[indenter],
                        "morphology_display": config.morphology_labels[morphology],
                        "baseline_magnitude": baseline_magnitude,
                        "optimized_magnitude": optimized_magnitude,
                        "magnitude_change_percent": magnitude_change,
                        "baseline_accuracy": baseline_accuracy,
                        "optimized_accuracy": optimized_accuracy,
                        "accuracy_change_pp": accuracy_change,
                        "status": "measured" if measured else "unavailable",
                    }
                )
    return output


def _calibration_training_sets(
    repetitions: tuple[int, ...], count: int, maximum: int, seed: int
) -> list[tuple[int, ...]]:
    choices = list(combinations(repetitions, count))
    if len(choices) <= maximum:
        return choices
    rng = np.random.default_rng(seed + count)
    selected = rng.choice(len(choices), size=maximum, replace=False)
    return [choices[int(index)] for index in np.sort(selected)]


def calibration_burden(
    samples: list[DecoderSample], config: PaperFigureConfig
) -> list[dict[str, Any]]:
    """Evaluate spatial decoding versus calibration repetitions per location."""

    repetitions = tuple(sorted({sample.repetition_index for sample in samples}))
    maximum_count = min(config.maximum_calibration_contacts, len(repetitions) - 1)
    output: list[dict[str, Any]] = []
    for count in range(1, maximum_count + 1):
        split_accuracy = []
        split_mae_mm = []
        test_samples_per_split: int | None = None
        for training_repetitions in _calibration_training_sets(
            repetitions, count, config.maximum_resamples, config.random_seed
        ):
            training_set = set(training_repetitions)
            templates = {
                hole: np.mean(
                    np.asarray(
                        [
                            sample.feature
                            for sample in samples
                            if sample.hole_index == hole
                            and sample.repetition_index in training_set
                        ]
                    ),
                    axis=0,
                )
                for hole in config.contact_positions_mm
            }
            testing = [
                sample
                for sample in samples
                if sample.repetition_index not in training_set
            ]
            predicted_holes = [
                _predict(sample.feature, templates)[0] for sample in testing
            ]
            correct = [
                prediction == sample.hole_index
                for prediction, sample in zip(predicted_holes, testing, strict=True)
            ]
            errors_mm = [
                abs(
                    config.contact_positions_mm[prediction]
                    - config.contact_positions_mm[sample.hole_index]
                )
                for prediction, sample in zip(predicted_holes, testing, strict=True)
            ]
            split_accuracy.append(_percent(correct))
            split_mae_mm.append(float(np.mean(errors_mm)))
            if test_samples_per_split is None:
                test_samples_per_split = len(testing)
            elif test_samples_per_split != len(testing):
                raise RuntimeError("calibration splits have unequal test-set sizes")
        output.append(
            {
                "material": samples[0].material,
                "material_display": config.material_labels[samples[0].material],
                "indenter": samples[0].indenter,
                "indenter_display": config.indenter_labels[samples[0].indenter],
                "morphology_id": samples[0].morphology,
                "morphology_display": config.morphology_labels[samples[0].morphology],
                "calibration_contacts_per_location": count,
                "split_count": len(split_accuracy),
                "test_accuracy_mean": float(np.mean(split_accuracy)),
                "test_accuracy_median": float(np.median(split_accuracy)),
                "test_accuracy_q25": float(np.percentile(split_accuracy, 25.0)),
                "test_accuracy_q75": float(np.percentile(split_accuracy, 75.0)),
                "test_samples_per_split": test_samples_per_split,
                "test_mae_mean_mm": float(np.mean(split_mae_mm)),
                "test_mae_median_mm": float(np.median(split_mae_mm)),
                "test_mae_q25_mm": float(np.percentile(split_mae_mm, 25.0)),
                "test_mae_q75_mm": float(np.percentile(split_mae_mm, 75.0)),
                "random_seed": config.random_seed,
                "status": "measured",
            }
        )
    return output


def _decoded_hdf5_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in values
        ]
    )


def _canonical_led_coordinates() -> np.ndarray:
    """Return distal-to-proximal LED coordinates over the physical full finger."""

    proximal_mm, distal_mm = TOTAL_Y_BOUNDS_MM
    distal_to_proximal = np.sort(np.asarray(LED_CENTERS_Y_MM, dtype=np.float64))[::-1]
    return (distal_mm - distal_to_proximal) / (distal_mm - proximal_mm)


def geometry_prior_zero_calibration(
    config: PaperFigureConfig,
    artifact_path: str | Path = DEFAULT_CONTACT_DATASET_H5,
    *,
    indenter: str = "sphere_10mm",
) -> list[dict[str, Any]]:
    """Evaluate the no-labelled-contact geometry-prior estimator at 5 N.

    The estimator uses each capture's unloaded canonical RGB reference and the
    known five-LED physical lattice. Hole positions enter only after inference
    to score absolute localization error.
    """

    artifact = _resolved_path(artifact_path)
    if not artifact.is_file():
        raise FileNotFoundError(f"missing compact contact dataset: {artifact}")
    optical_config = ContactLocalizationConfig(unloaded_frame_count=2)
    led_coordinates = _canonical_led_coordinates()
    output: list[dict[str, Any]] = []
    with h5py.File(artifact, "r") as data:
        if str(data.attrs.get("dataset_kind", "")) != "contact_dataset":
            raise ValueError(f"expected contact_dataset HDF5 artifact: {artifact}")
        session_material = _decoded_hdf5_strings(data["sessions/material"][:])
        session_morphology = _decoded_hdf5_strings(data["sessions/morphology"][:])
        frame_session = np.asarray(data["frames/session_index"][:], dtype=np.int64)
        frame_calibration = np.asarray(
            data["frames/calibration_index"][:], dtype=np.int64
        )
        frame_unloaded = np.asarray(data["frames/unloaded"][:], dtype=bool)
        frame_indenter = _decoded_hdf5_strings(data["frames/indenter"][:])
        frame_run = _decoded_hdf5_strings(data["frames/run_id"][:])
        frame_hole = np.asarray(data["frames/hole_index"][:], dtype=np.int64)
        frame_repetition = np.asarray(
            data["frames/repetition_index"][:], dtype=np.int64
        )
        frame_target = np.asarray(data["frames/target_force_n"][:], dtype=np.float64)

        reference_cache: dict[int, UnloadedOpticalReference] = {}

        def unloaded_reference(calibration_index: int) -> UnloadedOpticalReference:
            if calibration_index in reference_cache:
                return reference_cache[calibration_index]
            indices = np.flatnonzero(
                frame_unloaded & (frame_calibration == calibration_index)
            )
            if len(indices) != optical_config.unloaded_frame_count:
                raise RuntimeError(
                    f"calibration {calibration_index} requires exactly "
                    f"{optical_config.unloaded_frame_count} compact unloaded frames"
                )
            support = np.asarray(
                data["calibrations/support_mask"][calibration_index], dtype=bool
            )
            stored_reference = np.asarray(
                data["calibrations/reference_rgb"][calibration_index], dtype=np.uint8
            )
            frames = np.asarray(data["frames/rgb"][indices], dtype=np.uint8)
            frames[:, ~support] = stored_reference[~support]
            reference = build_unloaded_reference(frames, optical_config)
            reference_cache[calibration_index] = reference
            return reference

        for material in config.materials:
            for morphology in config.morphologies:
                session_indices = np.flatnonzero(
                    (session_material == material)
                    & (session_morphology == morphology)
                )
                if len(session_indices) != 1:
                    raise RuntimeError(
                        f"expected one compact session for {material}/{morphology}, "
                        f"found {len(session_indices)}"
                    )
                selected = np.flatnonzero(
                    (frame_session == int(session_indices[0]))
                    & ~frame_unloaded
                    & (frame_indenter == indenter)
                    & np.isclose(frame_target, config.high_force_n)
                )
                groups: dict[tuple[str, int, int], list[int]] = {}
                for index in selected:
                    key = (
                        str(frame_run[index]),
                        int(frame_hole[index]),
                        int(frame_repetition[index]),
                    )
                    groups.setdefault(key, []).append(int(index))
                errors_mm: list[float] = []
                for (_run_id, hole, _repetition), indices in sorted(groups.items()):
                    calibration_indices = np.unique(frame_calibration[indices])
                    if len(calibration_indices) != 1:
                        raise RuntimeError("one run maps to multiple unloaded captures")
                    calibration_index = int(calibration_indices[0])
                    reference = unloaded_reference(calibration_index)
                    current = np.rint(
                        np.median(data["frames/rgb"][indices], axis=0)
                    ).astype(np.uint8)
                    support = np.asarray(
                        data["calibrations/support_mask"][calibration_index], dtype=bool
                    )
                    current[~support] = reference.canonical_rgb[~support]
                    response, saturation_250, saturation_255 = (
                        compute_longitudinal_response(
                            current,
                            reference.canonical_rgb,
                            brightest_fraction=optical_config.brightest_fraction,
                            smoothing_sigma=optical_config.longitudinal_smoothing_sigma,
                        )
                    )
                    result = localize_response_profile(
                        response,
                        reference,
                        led_coordinates,
                        optical_config,
                        saturation_fraction_ge_250=saturation_250,
                        saturation_fraction_eq_255=saturation_255,
                    )
                    if not result.valid or result.position_mm is None:
                        raise RuntimeError(
                            f"zero-calibration estimate failed for "
                            f"{material}/{morphology}/{_run_id}: {result.status}"
                        )
                    ground_truth_mm = config.contact_positions_mm[hole]
                    errors_mm.append(abs(result.position_mm - ground_truth_mm))
                expected_count = len(config.contact_positions_mm) * 5
                if len(errors_mm) != expected_count:
                    raise RuntimeError(
                        f"expected {expected_count} zero-calibration samples for "
                        f"{material}/{morphology}/{indenter}, got {len(errors_mm)}"
                    )
                output.append(
                    {
                        "material": material,
                        "material_display": config.material_labels[material],
                        "indenter": indenter,
                        "indenter_display": config.indenter_labels[indenter],
                        "morphology_id": morphology,
                        "morphology_display": config.morphology_labels[morphology],
                        "calibration_contacts_per_location": 0,
                        "inference_regime": "geometry prior; no labelled contact calibration",
                        "sample_count": len(errors_mm),
                        "split_count": "",
                        "localization_mae_mean_mm": float(np.mean(errors_mm)),
                        "localization_mae_median_mm": float(np.median(errors_mm)),
                        "localization_mae_q25_mm": "",
                        "localization_mae_q75_mm": "",
                        "feature_definition": "5 N unloaded-relative Green response centroid",
                        "source_path": _display_path(artifact),
                        "status": "measured",
                    }
                )
    return output


def optional_calibration_rows(
    zero_calibration: list[dict[str, Any]],
    calibrated: list[dict[str, Any]],
    config: PaperFigureConfig,
    *,
    indenter: str = "sphere_10mm",
) -> list[dict[str, Any]]:
    """Combine the distinct zero-contact-calibration and calibrated regimes."""

    output = list(zero_calibration)
    for row in calibrated:
        if row["indenter"] != indenter:
            continue
        output.append(
            {
                "material": row["material"],
                "material_display": row["material_display"],
                "indenter": row["indenter"],
                "indenter_display": row["indenter_display"],
                "morphology_id": row["morphology_id"],
                "morphology_display": row["morphology_display"],
                "calibration_contacts_per_location": row[
                    "calibration_contacts_per_location"
                ],
                "inference_regime": "calibrated 6-region nearest-template",
                "sample_count": row["test_samples_per_split"],
                "split_count": row["split_count"],
                "localization_mae_mean_mm": row["test_mae_mean_mm"],
                "localization_mae_median_mm": row["test_mae_median_mm"],
                "localization_mae_q25_mm": row["test_mae_q25_mm"],
                "localization_mae_q75_mm": row["test_mae_q75_mm"],
                "feature_definition": "6-region signed Green change, 2 to 5 N",
                "source_path": "generated from configured longitudinal profiles",
                "status": row["status"],
            }
        )
    expected_rows = len(config.materials) * len(config.morphologies) * (
        config.maximum_calibration_contacts + 1
    )
    if len(output) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} optional-calibration rows, got {len(output)}"
        )
    return sorted(
        output,
        key=lambda row: (
            config.materials.index(str(row["material"])),
            config.morphologies.index(str(row["morphology_id"])),
            int(row["calibration_contacts_per_location"]),
        ),
    )


def load_spatial_distinguishability(
    config: PaperFigureConfig,
) -> list[dict[str, Any]]:
    """Load D_neighbor/W_contact without substituting within-contact cycle metrics."""

    output: list[dict[str, Any]] = []
    for material in config.materials:
        for indenter in config.indenters:
            for morphology in config.morphologies:
                root = config.source_root(material, morphology, indenter)
                path = root / "results" / "morphology_metrics.csv"
                source = None
                if path.is_file():
                    with path.open(newline="", encoding="utf-8") as stream:
                        for row in csv.DictReader(stream):
                            if (
                                row["material"] == material
                                and row["morphology"] == morphology
                                and row["indenter"] == indenter
                            ):
                                source = row
                                break
                if source is None:
                    message = (
                        f"missing spatial metric: {material}/{indenter}/{morphology}"
                    )
                    print(message, file=sys.stderr)
                    output.append(
                        {
                            "material": material,
                            "material_display": config.material_labels[material],
                            "indenter": indenter,
                            "indenter_display": config.indenter_labels[indenter],
                            "morphology_id": morphology,
                            "morphology_display": config.morphology_labels[morphology],
                            "D_neighbor_DN_per_N": "",
                            "W_contact_DN_per_N": "",
                            "Q_sep": "",
                            "W_cycle_DN": "",
                            "variability_definition": "independent re-contact",
                            "source_path": _display_path(path),
                            "status": "unavailable",
                        }
                    )
                    continue
                separation = float(source["D_neighbor_median_DN_per_N"])
                contact_variation = float(source["W_median_DN_per_N"])
                ratio = float(source["D_neighbor_over_W"])
                if (
                    separation <= 0.0
                    or contact_variation <= 0.0
                    or ratio <= 0.0
                    or not np.isclose(
                        ratio,
                        separation / contact_variation,
                        rtol=1.0e-9,
                        atol=1.0e-12,
                    )
                ):
                    raise ValueError(
                        f"invalid independent re-contact metric for "
                        f"{material}/{indenter}/{morphology}"
                    )
                output.append(
                    {
                        "material": material,
                        "material_display": config.material_labels[material],
                        "indenter": indenter,
                        "indenter_display": config.indenter_labels[indenter],
                        "morphology_id": morphology,
                        "morphology_display": config.morphology_labels[morphology],
                        "D_neighbor_DN_per_N": separation,
                        "W_contact_DN_per_N": contact_variation,
                        "Q_sep": ratio,
                        "W_cycle_DN": "",
                        "variability_definition": "independent re-contact",
                        "source_path": _display_path(path),
                        "status": "measured",
                    }
                )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_confusion_matrices(
    output_directory: Path,
    predictions: list[dict[str, Any]],
    config: PaperFigureConfig,
) -> None:
    for material in config.materials:
        for indenter in config.indenters:
            for morphology in config.morphologies:
                selected = [
                    row
                    for row in predictions
                    if row["material"] == material
                    and row["indenter"] == indenter
                    and row["morphology_id"] == morphology
                ]
                if not selected:
                    continue
                rows = []
                for true_hole in config.contact_positions_mm:
                    for predicted_hole in config.contact_positions_mm:
                        rows.append(
                            {
                                "material": material,
                                "indenter": indenter,
                                "morphology_id": morphology,
                                "true_hole_index": true_hole,
                                "true_contact_position_mm": config.contact_positions_mm[
                                    true_hole
                                ],
                                "predicted_hole_index": predicted_hole,
                                "predicted_contact_position_mm": config.contact_positions_mm[
                                    predicted_hole
                                ],
                                "count": sum(
                                    int(row["true_hole_index"] == true_hole)
                                    and int(
                                        row["predicted_hole_spatial"] == predicted_hole
                                    )
                                    for row in selected
                                ),
                            }
                        )
                name = f"fig5c_confusion_{material}_{indenter}_{morphology}.csv"
                _write_csv(output_directory / name, rows)


def run_analysis(config: PaperFigureConfig) -> dict[str, Path]:
    """Run the shared Figure 5(c)/6 analysis and write compact tables."""

    output = config.analysis_output_directory
    output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    for material in config.materials:
        for indenter in config.indenters:
            for morphology in config.morphologies:
                key = (material, indenter, morphology)
                samples, status = load_condition_samples(config, *key)
                if not samples:
                    print(
                        f"unavailable decoder condition {key}: {status}",
                        file=sys.stderr,
                    )
                    summaries.append(_missing_summary_row(config, *key, reason=status))
                    continue
                condition_predictions = decode_leave_one_repetition_out(samples, config)
                predictions.extend(condition_predictions)
                summaries.append(
                    _summarize_predictions(config, *key, condition_predictions)
                )
                calibration_rows.extend(calibration_burden(samples, config))
    _attach_baseline_comparisons(summaries, config)

    magnitude_rows = _baseline_relative_change_rows(summaries, config)
    scalar_spatial_rows = [
        {
            "material": row["material"],
            "material_display": row["material_display"],
            "indenter": row["indenter"],
            "indenter_display": row["indenter_display"],
            "morphology_id": row["morphology_id"],
            "morphology_display": row["morphology_display"],
            "scalar_only_accuracy_percent": row["accuracy_scalar_only"],
            "spatial_6_region_accuracy_percent": row["accuracy_exact"],
            "spatial_minus_scalar_percentage_points": (
                float(row["accuracy_exact"]) - float(row["accuracy_scalar_only"])
                if row["status"] == "measured"
                else ""
            ),
            "status": row["status"],
        }
        for row in summaries
    ]
    distinguishability_rows = load_spatial_distinguishability(config)
    zero_calibration_rows = geometry_prior_zero_calibration(config)
    optional_rows = optional_calibration_rows(
        zero_calibration_rows,
        calibration_rows,
        config,
    )

    summary_lookup = {
        (row["material"], row["indenter"], row["morphology_id"]): row
        for row in summaries
    }
    distinguishability_lookup = {
        (row["material"], row["indenter"], row["morphology_id"]): row
        for row in distinguishability_rows
    }
    magnitude_lookup = {
        (row["material"], row["indenter"], row["morphology"]): row
        for row in magnitude_rows
    }
    bundle_rows: list[dict[str, Any]] = []
    for key, row in summary_lookup.items():
        spatial = distinguishability_lookup[key]
        relative = magnitude_lookup.get(key)
        bundle_rows.append(
            {
                "record_type": "condition_summary",
                "material": row["material"],
                "material_display": row["material_display"],
                "indenter": row["indenter"],
                "indenter_display": row["indenter_display"],
                "morphology_id": row["morphology_id"],
                "morphology_display": row["morphology_display"],
                "calibration_contacts_per_location": "",
                "decoder_accuracy_percent": row["accuracy_exact"],
                "scalar_accuracy_percent": row["accuracy_scalar_only"],
                "optical_magnitude_median_dn": row["optical_magnitude_median_dn"],
                "magnitude_change_percent": ""
                if relative is None
                else relative["magnitude_change_percent"],
                "accuracy_change_pp": ""
                if relative is None
                else relative["accuracy_change_pp"],
                "D_neighbor_DN_per_N": spatial["D_neighbor_DN_per_N"],
                "W_contact_DN_per_N": spatial["W_contact_DN_per_N"],
                "Q_sep": spatial["Q_sep"],
                "calibration_accuracy_mean_percent": "",
                "decoder_status": row["status"],
                "spatial_metric_status": spatial["status"],
                "status": (
                    "measured"
                    if row["status"] == "measured" and spatial["status"] == "measured"
                    else "partially unavailable"
                ),
            }
        )
    for row in calibration_rows:
        bundle_rows.append(
            {
                "record_type": "calibration_curve",
                "material": row["material"],
                "material_display": row["material_display"],
                "indenter": row["indenter"],
                "indenter_display": row["indenter_display"],
                "morphology_id": row["morphology_id"],
                "morphology_display": row["morphology_display"],
                "calibration_contacts_per_location": row[
                    "calibration_contacts_per_location"
                ],
                "decoder_accuracy_percent": "",
                "scalar_accuracy_percent": "",
                "optical_magnitude_median_dn": "",
                "magnitude_change_percent": "",
                "accuracy_change_pp": "",
                "D_neighbor_DN_per_N": "",
                "W_contact_DN_per_N": "",
                "Q_sep": "",
                "calibration_accuracy_mean_percent": row["test_accuracy_mean"],
                "decoder_status": row["status"],
                "spatial_metric_status": "not applicable",
                "status": row["status"],
            }
        )

    paths = {
        "accuracy": _write_csv(output / "fig5c_accuracy_summary.csv", summaries),
        "predictions": _write_csv(
            output / "fig5c_per_sample_predictions.csv", predictions
        ),
        "magnitude": _write_csv(
            output / "fig6a_magnitude_vs_accuracy.csv", magnitude_rows
        ),
        "distinguishability": _write_csv(
            output / "fig6b_spatial_distinguishability.csv",
            distinguishability_rows,
        ),
        "scalar_spatial": _write_csv(
            output / "fig6c_scalar_vs_spatial.csv", scalar_spatial_rows
        ),
        "optional_calibration": _write_csv(
            output / "fig6c_optional_calibration_10mm_combined.csv",
            optional_rows,
        ),
        "calibration": _write_csv(
            output / "fig6d_calibration_burden.csv", calibration_rows
        ),
        "bundle": _write_csv(output / "fig56_summary_bundle.csv", bundle_rows),
    }
    _write_confusion_matrices(output, predictions, config)
    return paths


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one generated summary table."""

    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    for name, path in run_analysis(config).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

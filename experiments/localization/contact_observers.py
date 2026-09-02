"""Contact inference from already-extracted optical representations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .optical_features import (
    DenseProfileConfig,
    mean_center_l2,
    robust_zscore,
)


FEATURE_NOISE_FLOOR_DN = 0.75
CONTACT_Z_THRESHOLD = 4.0
_NORMALIZATION_MODES = ("none", "mean_center_l2", "robust_zscore")


@dataclass(frozen=True)
class ContactEstimate:
    """Baseline-relative LED response and its response-weighted position."""

    response: np.ndarray
    contact_detected: bool
    predicted_led_index: int
    position_mm: float | None
    top_two_margin: float

    def __post_init__(self) -> None:
        response = np.asarray(self.response, dtype=np.float64)
        if response.ndim != 1 or not len(response) or not np.all(np.isfinite(response)):
            raise ValueError("response must be a finite nonempty vector")
        if not isinstance(self.contact_detected, bool):
            raise ValueError("contact_detected must be a bool")
        if not 0 <= self.predicted_led_index < len(response):
            raise ValueError("predicted_led_index is outside the response vector")
        if self.position_mm is not None and not np.isfinite(self.position_mm):
            raise ValueError("position_mm must be finite when available")
        if not np.isfinite(self.top_two_margin):
            raise ValueError("top_two_margin must be finite")
        response = response.copy()
        response.setflags(write=False)
        object.__setattr__(self, "response", response)


@dataclass(frozen=True)
class DenseTemplateModel:
    """Position-labelled canonical profiles and their extraction metadata."""

    positions_mm: np.ndarray
    templates: np.ndarray
    canonical_shape: tuple[int, int]
    feature_config: DenseProfileConfig
    normalization: str = "mean_center_l2"

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_mm, dtype=np.float64)
        templates = np.asarray(self.templates, dtype=np.float64)
        if (
            positions.ndim != 1
            or not len(positions)
            or not np.all(np.isfinite(positions))
            or not np.all(np.diff(positions) > 0.0)
        ):
            raise ValueError("positions_mm must be finite, nonempty, and increasing")
        if (
            templates.ndim != 2
            or templates.shape[0] != len(positions)
            or not np.all(np.isfinite(templates))
        ):
            raise ValueError("templates must be a finite positions x features array")
        if (
            len(self.canonical_shape) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in self.canonical_shape
            )
            or templates.shape[1] != self.canonical_shape[0]
        ):
            raise ValueError("canonical_shape must match the template profile length")
        if self.normalization not in _NORMALIZATION_MODES:
            raise ValueError(
                f"normalization must be one of {', '.join(_NORMALIZATION_MODES)}"
            )
        positions = positions.copy()
        templates = templates.copy()
        positions.setflags(write=False)
        templates.setflags(write=False)
        object.__setattr__(self, "positions_mm", positions)
        object.__setattr__(self, "templates", templates)


@dataclass(frozen=True)
class DensePositionEstimate:
    """Conditional optical position and all normalized correlation scores."""

    position_mm: float
    matched_index: int
    similarities: np.ndarray

    def __post_init__(self) -> None:
        similarities = np.asarray(self.similarities, dtype=np.float64)
        if similarities.ndim != 1 or not np.all(np.isfinite(similarities)):
            raise ValueError("similarities must be a finite vector")
        if not 0 <= self.matched_index < len(similarities):
            raise ValueError("matched_index is outside the similarity vector")
        if not np.isfinite(self.position_mm):
            raise ValueError("position_mm must be finite")
        similarities = similarities.copy()
        similarities.setflags(write=False)
        object.__setattr__(self, "similarities", similarities)


@dataclass(frozen=True)
class AffineCentroidModel:
    """Physical position as an affine function of normalized response centroid."""

    slope_mm: float
    intercept_mm: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.slope_mm) or not np.isfinite(self.intercept_mm):
            raise ValueError("affine centroid coefficients must be finite")


def _normalize(profile: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("profile must be a finite nonempty vector")
    if mode == "none":
        return values.copy()
    if mode == "mean_center_l2":
        return mean_center_l2(values)
    if mode == "robust_zscore":
        return robust_zscore(values)
    raise ValueError(f"normalization must be one of {', '.join(_NORMALIZATION_MODES)}")


def unloaded_baseline_statistics(
    feature_samples: np.ndarray,
    *,
    noise_floor_dn: float = FEATURE_NOISE_FLOOR_DN,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-channel temporal medians and robust unloaded noise scales."""

    samples = np.asarray(feature_samples, dtype=np.float64)
    if (
        samples.ndim != 2
        or not samples.shape[1]
        or not len(samples)
        or not np.all(np.isfinite(samples))
    ):
        raise ValueError("feature_samples must be a finite nonempty 2-D array")
    if not np.isfinite(noise_floor_dn) or noise_floor_dn <= 0.0:
        raise ValueError("noise_floor_dn must be finite and positive")
    baseline = np.median(samples, axis=0)
    median_absolute_deviation = np.median(np.abs(samples - baseline), axis=0)
    noise_sigma = np.maximum(1.4826 * median_absolute_deviation, noise_floor_dn)
    return baseline, noise_sigma


def estimate_contact_position(
    features: np.ndarray,
    unloaded_baseline: np.ndarray,
    unloaded_noise_sigma: np.ndarray,
    led_positions_mm: np.ndarray,
) -> ContactEstimate:
    """Estimate noise-gated contact from baseline-relative LED response."""

    current = np.asarray(features, dtype=np.float64)
    baseline = np.asarray(unloaded_baseline, dtype=np.float64)
    noise_sigma = np.asarray(unloaded_noise_sigma, dtype=np.float64)
    positions = np.asarray(led_positions_mm, dtype=np.float64)
    if current.ndim != 1 or len(current) < 2:
        raise ValueError("features must contain at least two channels")
    for name, values in (
        ("features", current),
        ("unloaded_baseline", baseline),
        ("unloaded_noise_sigma", noise_sigma),
        ("led_positions_mm", positions),
    ):
        if values.shape != current.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must match the finite feature vector")
    position_steps = np.diff(positions)
    if not (np.all(position_steps > 0.0) or np.all(position_steps < 0.0)):
        raise ValueError("led_positions_mm must be strictly ordered")
    if np.any(noise_sigma < 0.0):
        raise ValueError("unloaded_noise_sigma must be nonnegative")

    response = current - baseline
    standardized_positive = np.maximum(response, 0.0) / np.maximum(
        noise_sigma,
        FEATURE_NOISE_FLOOR_DN,
    )
    contact_detected = bool(np.max(standardized_positive) >= CONTACT_Z_THRESHOLD)
    predicted = int(np.argmax(response))
    sorted_response = np.sort(response)
    margin = float(sorted_response[-1] - sorted_response[-2])
    positive = np.maximum(response, 0.0)
    total_positive = float(np.sum(positive))
    position = None
    if contact_detected and total_positive > np.finfo(np.float64).eps:
        position = float(np.dot(positive, positions) / total_positive)
    return ContactEstimate(
        response=response,
        contact_detected=contact_detected,
        predicted_led_index=predicted,
        position_mm=position,
        top_two_margin=margin,
    )


def build_dense_template_model(
    profiles: np.ndarray,
    positions_mm: np.ndarray,
    *,
    aggregation: str = "median",
    canonical_shape: tuple[int, int] | None = None,
    feature_config: DenseProfileConfig | None = None,
    normalization: str = "mean_center_l2",
) -> DenseTemplateModel:
    """Aggregate normalized profiles at each labelled physical position."""

    values = np.asarray(profiles, dtype=np.float64)
    positions = np.asarray(positions_mm, dtype=np.float64)
    if (
        values.ndim != 2
        or not len(values)
        or not np.all(np.isfinite(values))
        or positions.shape != (len(values),)
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("profiles and positions_mm must be finite aligned arrays")
    if aggregation != "median":
        raise ValueError("aggregation currently supports only 'median'")
    if normalization not in _NORMALIZATION_MODES:
        raise ValueError(
            f"normalization must be one of {', '.join(_NORMALIZATION_MODES)}"
        )

    normalized = np.vstack([_normalize(profile, normalization) for profile in values])
    unique_positions = np.unique(positions)
    templates = np.vstack(
        [np.median(normalized[positions == position], axis=0) for position in unique_positions]
    )
    templates = np.vstack([_normalize(template, normalization) for template in templates])
    if canonical_shape is None:
        canonical_shape = (values.shape[1], 1)
    return DenseTemplateModel(
        positions_mm=unique_positions,
        templates=templates,
        canonical_shape=canonical_shape,
        feature_config=feature_config or DenseProfileConfig(),
        normalization=normalization,
    )


def estimate_dense_template_position(
    profile: np.ndarray,
    model: DenseTemplateModel,
) -> DensePositionEstimate:
    """Return the optical position conditional on an external contact gate."""

    normalized = _normalize(profile, model.normalization)
    if normalized.shape != (model.templates.shape[1],):
        raise ValueError("profile length does not match the dense template model")
    if model.normalization == "mean_center_l2":
        similarities = model.templates @ normalized
    else:
        template_norms = np.linalg.norm(model.templates, axis=1)
        profile_norm = float(np.linalg.norm(normalized))
        if profile_norm <= np.finfo(np.float64).eps or np.any(
            template_norms <= np.finfo(np.float64).eps
        ):
            raise ValueError("template correlation requires nonconstant profiles")
        similarities = (model.templates @ normalized) / (template_norms * profile_norm)
    matched_index = int(np.argmax(similarities))
    return DensePositionEstimate(
        position_mm=float(model.positions_mm[matched_index]),
        matched_index=matched_index,
        similarities=similarities,
    )


def response_centroid(response_profile: np.ndarray) -> float:
    """Return the robust response centroid in normalized longitudinal units."""

    response = np.asarray(response_profile, dtype=np.float64)
    if response.ndim != 1 or len(response) < 2 or not np.all(np.isfinite(response)):
        raise ValueError("response_profile must be a finite vector of length >= 2")
    positive = np.maximum(response - np.percentile(response, 10.0), 0.0)
    threshold = float(np.percentile(positive, 60.0))
    weights = np.maximum(positive - threshold, 0.0)
    total = float(np.sum(weights))
    if total <= np.finfo(np.float64).eps:
        raise ValueError("response_profile has no robust positive response")
    coordinates = np.linspace(0.0, 1.0, len(response))
    return float(np.dot(weights, coordinates) / total)


def fit_affine_position_from_centroid(
    centroids: np.ndarray,
    positions_mm: np.ndarray,
) -> AffineCentroidModel:
    """Fit physical position from labelled normalized response centroids."""

    centroid_values = np.asarray(centroids, dtype=np.float64)
    positions = np.asarray(positions_mm, dtype=np.float64)
    if (
        centroid_values.ndim != 1
        or positions.shape != centroid_values.shape
        or len(centroid_values) < 2
        or not np.all(np.isfinite(centroid_values))
        or not np.all(np.isfinite(positions))
        or np.ptp(centroid_values) <= np.finfo(np.float64).eps
    ):
        raise ValueError("centroids and positions_mm must define a finite affine fit")
    design = np.column_stack((centroid_values, np.ones(len(centroid_values))))
    slope, intercept = np.linalg.lstsq(design, positions, rcond=None)[0]
    return AffineCentroidModel(float(slope), float(intercept))


def estimate_affine_position_from_centroid(
    profile: np.ndarray,
    unloaded_reference: np.ndarray,
    model: AffineCentroidModel,
) -> float:
    """Estimate position from an explicitly baseline-relative dense response."""

    values = np.asarray(profile, dtype=np.float64)
    baseline = np.asarray(unloaded_reference, dtype=np.float64)
    if values.ndim != 1 or baseline.shape != values.shape:
        raise ValueError("profile and unloaded_reference must be equal vectors")
    centroid = response_centroid(values - baseline)
    return model.slope_mm * centroid + model.intercept_mm


def save_dense_template_model(path: str | Path, model: DenseTemplateModel) -> None:
    """Save a dense template model as non-pickle compressed NumPy arrays."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    config = asdict(model.feature_config)
    np.savez_compressed(
        destination,
        positions_mm=model.positions_mm,
        templates=model.templates,
        canonical_shape=np.asarray(model.canonical_shape, dtype=np.int64),
        normalization=np.asarray(model.normalization),
        **{f"feature_{name}": np.asarray(value) for name, value in config.items()},
    )


def load_dense_template_model(path: str | Path) -> DenseTemplateModel:
    """Load a dense template model without enabling NumPy pickle support."""

    with np.load(Path(path), allow_pickle=False) as data:
        config = DenseProfileConfig(
            mode=str(data["feature_mode"].item()),
            transverse_start_fraction=float(
                data["feature_transverse_start_fraction"].item()
            ),
            transverse_stop_fraction=float(
                data["feature_transverse_stop_fraction"].item()
            ),
            transverse_reduction=(
                str(data["feature_transverse_reduction"].item())
                if "feature_transverse_reduction" in data
                else "top_fraction"
            ),
            top_fraction=float(data["feature_top_fraction"].item()),
            longitudinal_smoothing_sigma_px=float(
                data["feature_longitudinal_smoothing_sigma_px"].item()
            ),
            highpass_sigma_px=float(data["feature_highpass_sigma_px"].item()),
        )
        return DenseTemplateModel(
            positions_mm=data["positions_mm"],
            templates=data["templates"],
            canonical_shape=tuple(
                int(value) for value in data["canonical_shape"].tolist()
            ),
            feature_config=config,
            normalization=str(data["normalization"].item()),
        )


__all__ = [
    "CONTACT_Z_THRESHOLD",
    "FEATURE_NOISE_FLOOR_DN",
    "AffineCentroidModel",
    "ContactEstimate",
    "DensePositionEstimate",
    "DenseTemplateModel",
    "build_dense_template_model",
    "estimate_affine_position_from_centroid",
    "estimate_contact_position",
    "estimate_dense_template_position",
    "fit_affine_position_from_centroid",
    "load_dense_template_model",
    "response_centroid",
    "save_dense_template_model",
    "unloaded_baseline_statistics",
]

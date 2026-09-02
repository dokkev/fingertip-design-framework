"""Pure optical feature extraction on LED ROIs and canonical finger images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


_DENSE_PROFILE_MODES = (
    "top10_red",
    "mean_red",
    "abs_highpass_red",
    "red_gradient",
    "red_fraction",
)


@dataclass(frozen=True)
class DenseProfileConfig:
    """Parameters for one longitudinal profile from a canonical RGB image."""

    mode: str = "top10_red"
    transverse_start_fraction: float = 0.0
    transverse_stop_fraction: float = 1.0
    top_fraction: float = 0.10
    longitudinal_smoothing_sigma_px: float = 0.0
    highpass_sigma_px: float = 5.0

    def __post_init__(self) -> None:
        if self.mode not in _DENSE_PROFILE_MODES:
            raise ValueError(
                f"mode must be one of {', '.join(_DENSE_PROFILE_MODES)}"
            )
        if not (
            0.0 <= self.transverse_start_fraction
            < self.transverse_stop_fraction
            <= 1.0
        ):
            raise ValueError("transverse fractions must define a nonempty subset")
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        for name, value in (
            (
                "longitudinal_smoothing_sigma_px",
                self.longitudinal_smoothing_sigma_px,
            ),
            ("highpass_sigma_px", self.highpass_sigma_px),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.mode == "abs_highpass_red" and self.highpass_sigma_px == 0.0:
            raise ValueError("highpass_sigma_px must be positive for abs_highpass_red")


def _canonical_rgb(canonical_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(canonical_rgb)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or min(image.shape[:2]) < 2
    ):
        raise ValueError("canonical_rgb must be an H x W x 3 uint8 array")
    return image


def _transverse_slice(width: int, config: DenseProfileConfig) -> slice:
    start = min(width - 1, int(np.floor(config.transverse_start_fraction * width)))
    stop = max(start + 1, int(np.ceil(config.transverse_stop_fraction * width)))
    return slice(start, min(width, stop))


def _brightest_fraction_profile(values: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(np.ceil(fraction * values.shape[1])))
    start = values.shape[1] - count
    return np.mean(np.partition(values, start, axis=1)[:, start:], axis=1)


def _smooth_longitudinal(
    profile: np.ndarray,
    sigma_px: float,
) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float64)
    if sigma_px > 0.0:
        values = cv2.GaussianBlur(values[:, None], (1, 0), sigma_px).ravel()
    if not np.all(np.isfinite(values)):
        raise RuntimeError("dense optical profile is not finite")
    return values


def extract_dense_profile(
    canonical_rgb: np.ndarray,
    config: DenseProfileConfig,
) -> np.ndarray:
    """Extract one longitudinal optical profile from canonical RGB pixels."""

    image = _canonical_rgb(canonical_rgb)
    columns = _transverse_slice(image.shape[1], config)
    red = image[:, :, 0].astype(np.float32)
    selected_red = red[:, columns]

    if config.mode == "top10_red":
        profile = _brightest_fraction_profile(selected_red, config.top_fraction)
    elif config.mode == "mean_red":
        profile = np.mean(selected_red, axis=1)
    elif config.mode == "abs_highpass_red":
        smooth = cv2.GaussianBlur(
            red,
            (0, 0),
            config.highpass_sigma_px,
            config.highpass_sigma_px,
        )
        profile = _brightest_fraction_profile(
            np.abs(red[:, columns] - smooth[:, columns]),
            config.top_fraction,
        )
    elif config.mode == "red_gradient":
        gradient = cv2.Sobel(red, cv2.CV_32F, 0, 1, ksize=3)
        profile = np.mean(np.abs(gradient[:, columns]), axis=1)
    else:
        rgb = image[:, columns].astype(np.float32)
        denominator = np.sum(rgb, axis=2) + np.finfo(np.float32).eps
        profile = np.mean(rgb[:, :, 0] / denominator, axis=1)

    profile = _smooth_longitudinal(
        profile,
        config.longitudinal_smoothing_sigma_px,
    )
    if profile.shape != (image.shape[0],) or not np.all(np.isfinite(profile)):
        raise RuntimeError("dense optical profile is not finite")
    return profile


def extract_dense_response_profile(
    canonical_rgb: np.ndarray,
    unloaded_canonical_rgb: np.ndarray,
    config: DenseProfileConfig,
) -> np.ndarray:
    """Extract a dense feature after explicit unloaded-image subtraction."""

    image = _canonical_rgb(canonical_rgb)
    unloaded = _canonical_rgb(unloaded_canonical_rgb)
    if unloaded.shape != image.shape:
        raise ValueError("canonical_rgb and unloaded_canonical_rgb must match")
    if config.mode != "abs_highpass_red":
        raise ValueError(
            "unloaded-relative extraction currently supports abs_highpass_red"
        )
    columns = _transverse_slice(image.shape[1], config)
    red_difference = (
        image[:, :, 0].astype(np.float32)
        - unloaded[:, :, 0].astype(np.float32)
    )
    smooth = cv2.GaussianBlur(
        red_difference,
        (0, 0),
        config.highpass_sigma_px,
        config.highpass_sigma_px,
    )
    response = np.abs(red_difference[:, columns] - smooth[:, columns])
    profile = _brightest_fraction_profile(response, config.top_fraction)
    return _smooth_longitudinal(
        profile,
        config.longitudinal_smoothing_sigma_px,
    )


def mean_center_l2(profile: np.ndarray) -> np.ndarray:
    """Mean-center and normalize one profile to unit Euclidean norm."""

    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("profile must be a finite nonempty vector")
    centered = values - np.mean(values)
    norm = float(np.linalg.norm(centered))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("profile has no mean-centered variation")
    return centered / norm


def robust_zscore(profile: np.ndarray) -> np.ndarray:
    """Median-center one profile and divide by its robust MAD scale."""

    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("profile must be a finite nonempty vector")
    median = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - median)))
    if scale <= np.finfo(np.float64).eps:
        raise ValueError("profile has no robust variation")
    return (values - median) / scale


__all__ = [
    "DenseProfileConfig",
    "extract_dense_profile",
    "extract_dense_response_profile",
    "mean_center_l2",
    "robust_zscore",
]

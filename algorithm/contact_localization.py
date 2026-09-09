"""Calibration-free contact localization from registered green-channel response."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .canonical import (
    CanonicalFingerConfig,
    CanonicalFingerMap,
    build_canonical_map,
    landmark_longitudinal_coordinates,
    transform_canonical_map,
    warp_to_canonical,
)
from .fingertip_segmentation import FingertipRegion, segment_fingertip
from .led_localization import (
    LED_POSITIONS_MM,
    LedGeometry,
    detect_leds,
    estimate_led_similarity_transform,
    reanchor_leds,
    track_leds,
)


@dataclass(frozen=True)
class ContactLocalizationConfig:
    """Fixed optical-reduction and contact-gating parameters."""

    unloaded_frame_count: int = 30
    brightest_fraction: float = 0.10
    longitudinal_smoothing_sigma: float = 2.0
    temporal_median_window: int = 3
    noise_sigma_multiplier: float = 4.0
    noise_floor_dn: float = 0.75
    minimum_total_evidence_dn: float = 4.0
    minimum_peak_to_background_ratio: float = 1.20
    maximum_normalized_entropy: float = 0.985
    maximum_saturation_fraction_ge_250: float = 0.25
    maximum_saturation_fraction_eq_255: float = 0.05
    maximum_extrapolation_mm: float = 11.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.unloaded_frame_count, int)
            or isinstance(self.unloaded_frame_count, bool)
            or self.unloaded_frame_count < 2
        ):
            raise ValueError("unloaded_frame_count must be an integer >= 2")
        if not 0.0 < self.brightest_fraction <= 1.0:
            raise ValueError("brightest_fraction must be in (0, 1]")
        if self.longitudinal_smoothing_sigma < 0.0:
            raise ValueError("longitudinal_smoothing_sigma must be nonnegative")
        if (
            not isinstance(self.temporal_median_window, int)
            or isinstance(self.temporal_median_window, bool)
            or self.temporal_median_window < 1
            or self.temporal_median_window % 2 == 0
        ):
            raise ValueError("temporal_median_window must be a positive odd integer")
        for name in (
            "noise_sigma_multiplier",
            "noise_floor_dn",
            "minimum_total_evidence_dn",
            "minimum_peak_to_background_ratio",
            "maximum_extrapolation_mm",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.maximum_normalized_entropy <= 1.0:
            raise ValueError("maximum_normalized_entropy must be in (0, 1]")
        for name in (
            "maximum_saturation_fraction_ge_250",
            "maximum_saturation_fraction_eq_255",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class UnloadedOpticalReference:
    """Registered unloaded image and per-row robust response statistics."""

    canonical_rgb: np.ndarray
    response_center_dn: np.ndarray
    response_sigma_dn: np.ndarray
    response_threshold_dn: np.ndarray
    frame_count: int

    def __post_init__(self) -> None:
        image = np.asarray(self.canonical_rgb, dtype=np.float32)
        center = np.asarray(self.response_center_dn, dtype=np.float64)
        sigma = np.asarray(self.response_sigma_dn, dtype=np.float64)
        threshold = np.asarray(self.response_threshold_dn, dtype=np.float64)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("canonical_rgb must be an H x W x 3 array")
        if center.shape != (image.shape[0],):
            raise ValueError("response statistics must match canonical longitudinal rows")
        if sigma.shape != center.shape or threshold.shape != center.shape:
            raise ValueError("response statistics must have equal shape")
        if not all(np.all(np.isfinite(value)) for value in (image, center, sigma, threshold)):
            raise ValueError("unloaded reference arrays must be finite")
        if np.any(sigma <= 0.0) or np.any(threshold < center):
            raise ValueError("unloaded noise must be positive and thresholds valid")
        if self.frame_count < 2:
            raise ValueError("unloaded reference requires at least two frames")
        image = image.copy()
        center = center.copy()
        sigma = sigma.copy()
        threshold = threshold.copy()
        for value in (image, center, sigma, threshold):
            value.setflags(write=False)
        object.__setattr__(self, "canonical_rgb", image)
        object.__setattr__(self, "response_center_dn", center)
        object.__setattr__(self, "response_sigma_dn", sigma)
        object.__setattr__(self, "response_threshold_dn", threshold)


@dataclass(frozen=True)
class ContactLocalizationResult:
    """One contact decision, physical estimate, and interpretable quality metrics."""

    valid: bool
    status: str
    contact_detected: bool
    position_mm: float | None
    normalized_position: float | None
    total_evidence_dn: float
    peak_snr: float
    peak_to_background_ratio: float
    normalized_spatial_entropy: float
    temporal_consistency_dn: float
    saturation_fraction_ge_250: float
    saturation_fraction_eq_255: float
    response_profile: np.ndarray
    evidence_profile: np.ndarray

    def __post_init__(self) -> None:
        response = np.asarray(self.response_profile, dtype=np.float64)
        evidence = np.asarray(self.evidence_profile, dtype=np.float64)
        if response.ndim != 1 or evidence.shape != response.shape:
            raise ValueError("response and evidence profiles must be equal 1-D arrays")
        if not np.all(np.isfinite(response)) or not np.all(np.isfinite(evidence)):
            raise ValueError("response and evidence profiles must be finite")
        if self.valid and self.contact_detected:
            if self.position_mm is None or self.normalized_position is None:
                raise ValueError("a valid detected contact requires a position")
        response = response.copy()
        evidence = evidence.copy()
        response.setflags(write=False)
        evidence.setflags(write=False)
        object.__setattr__(self, "response_profile", response)
        object.__setattr__(self, "evidence_profile", evidence)


def _validate_canonical_rgb(canonical_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(canonical_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("canonical_rgb must be an H x W x 3 uint8 array")
    return image


def _smooth_profile(profile: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float64)
    if sigma == 0.0:
        return values.copy()
    return cv2.GaussianBlur(
        values[:, None],
        (1, 0),
        sigmaX=0.0,
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REPLICATE,
    ).ravel()


def compute_longitudinal_response(
    canonical_rgb: np.ndarray,
    unloaded_rgb: np.ndarray,
    *,
    brightest_fraction: float = 0.10,
    smoothing_sigma: float = 2.0,
) -> tuple[np.ndarray, float, float]:
    """Reduce positive green change to one longitudinal response profile.

    Returns the response and saturation fractions of the transverse pixels
    selected by the same brightest-fraction reduction.
    """

    current = _validate_canonical_rgb(canonical_rgb)
    unloaded = np.asarray(unloaded_rgb)
    if unloaded.shape != current.shape:
        raise ValueError("unloaded_rgb must match canonical_rgb")
    if not np.issubdtype(unloaded.dtype, np.number):
        raise ValueError("unloaded_rgb must be numeric")
    if not 0.0 < brightest_fraction <= 1.0:
        raise ValueError("brightest_fraction must be in (0, 1]")
    if smoothing_sigma < 0.0:
        raise ValueError("smoothing_sigma must be nonnegative")
    green = current[:, :, 1].astype(np.float32)
    unloaded_green = unloaded[:, :, 1].astype(np.float32)
    positive = np.maximum(green - unloaded_green, 0.0)
    selected_count = max(1, int(np.ceil(brightest_fraction * positive.shape[1])))
    selected_indices = np.argpartition(
        positive,
        positive.shape[1] - selected_count,
        axis=1,
    )[:, -selected_count:]
    selected_response = np.take_along_axis(positive, selected_indices, axis=1)
    selected_green = np.take_along_axis(green, selected_indices, axis=1)
    response = _smooth_profile(selected_response.mean(axis=1), smoothing_sigma)
    return (
        response,
        float(np.mean(selected_green >= 250.0)),
        float(np.mean(selected_green >= 255.0)),
    )


def build_unloaded_reference(
    canonical_frames: Iterable[np.ndarray],
    config: ContactLocalizationConfig = ContactLocalizationConfig(),
) -> UnloadedOpticalReference:
    """Build a temporal-median unloaded reference with a nonzero noise floor."""

    frames = [_validate_canonical_rgb(frame) for frame in canonical_frames]
    if len(frames) < config.unloaded_frame_count:
        raise ValueError(
            f"need at least {config.unloaded_frame_count} unloaded frames, got {len(frames)}"
        )
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError("all unloaded canonical frames must have equal shape")
    stack = np.stack(frames, axis=0)
    canonical_median = np.median(stack, axis=0).astype(np.float32)
    profiles = np.stack(
        [
            compute_longitudinal_response(
                frame,
                canonical_median,
                brightest_fraction=config.brightest_fraction,
                smoothing_sigma=config.longitudinal_smoothing_sigma,
            )[0]
            for frame in frames
        ],
        axis=0,
    )
    center = np.median(profiles, axis=0)
    mad = np.median(np.abs(profiles - center[None, :]), axis=0)
    sigma = np.maximum(1.4826 * mad, config.noise_floor_dn)
    threshold = center + config.noise_sigma_multiplier * sigma
    return UnloadedOpticalReference(
        canonical_rgb=canonical_median,
        response_center_dn=center,
        response_sigma_dn=sigma,
        response_threshold_dn=threshold,
        frame_count=len(frames),
    )


def causal_median_response(
    response_history: Iterable[np.ndarray],
    window: int = 3,
) -> np.ndarray:
    """Return the causal median of the newest odd-length response window."""

    if not isinstance(window, int) or isinstance(window, bool) or window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    history = [np.asarray(profile, dtype=np.float64) for profile in response_history]
    if not history:
        raise ValueError("response_history must not be empty")
    selected = history[-window:]
    if any(profile.ndim != 1 or profile.shape != selected[0].shape for profile in selected):
        raise ValueError("response profiles must be equal 1-D arrays")
    return np.median(np.stack(selected, axis=0), axis=0)


def canonical_position_to_mm(
    normalized_position: float,
    led_coordinates: np.ndarray,
    led_positions_mm: np.ndarray = LED_POSITIONS_MM,
    *,
    maximum_extrapolation_mm: float = 11.0,
) -> float:
    """Map a canonical coordinate through the known 11-mm LED lattice."""

    coordinate = float(normalized_position)
    anchors = np.asarray(led_coordinates, dtype=np.float64)
    positions = np.asarray(led_positions_mm, dtype=np.float64)
    if anchors.shape != positions.shape or anchors.ndim != 1 or len(anchors) < 2:
        raise ValueError("LED coordinate and physical arrays must have equal length >= 2")
    if not np.isfinite(coordinate) or not np.all(np.isfinite(anchors)):
        raise ValueError("canonical coordinates must be finite")
    if not np.all(np.diff(anchors) > 0.0) or not np.all(np.diff(positions) > 0.0):
        raise ValueError("LED coordinates must be strictly distal-to-proximal")
    physical = float(np.interp(coordinate, anchors, positions))
    if coordinate < anchors[0]:
        slope = (positions[1] - positions[0]) / (anchors[1] - anchors[0])
        physical = float(positions[0] + slope * (coordinate - anchors[0]))
        if positions[0] - physical > maximum_extrapolation_mm:
            raise ValueError("contact lies beyond the allowed distal extrapolation")
    elif coordinate > anchors[-1]:
        slope = (positions[-1] - positions[-2]) / (anchors[-1] - anchors[-2])
        physical = float(positions[-1] + slope * (coordinate - anchors[-1]))
        if physical - positions[-1] > maximum_extrapolation_mm:
            raise ValueError("contact lies beyond the allowed proximal extrapolation")
    return physical


def localize_response_profile(
    response_profile: np.ndarray,
    unloaded_reference: UnloadedOpticalReference,
    led_coordinates: np.ndarray,
    config: ContactLocalizationConfig = ContactLocalizationConfig(),
    *,
    temporal_consistency_dn: float = 0.0,
    saturation_fraction_ge_250: float = 0.0,
    saturation_fraction_eq_255: float = 0.0,
) -> ContactLocalizationResult:
    """Estimate continuous contact position from one registered response profile."""

    response = np.asarray(response_profile, dtype=np.float64)
    if response.shape != unloaded_reference.response_center_dn.shape:
        raise ValueError("response_profile must match the unloaded reference")
    if not np.all(np.isfinite(response)):
        raise ValueError("response_profile must be finite")
    evidence = np.maximum(response - unloaded_reference.response_threshold_dn, 0.0)
    total = float(np.sum(evidence))
    snr = (response - unloaded_reference.response_center_dn) / unloaded_reference.response_sigma_dn
    peak_snr = float(np.max(snr))
    background = float(np.median(response))
    peak = float(np.max(response))
    ratio_floor = config.noise_floor_dn
    peak_to_background = (peak + ratio_floor) / (background + ratio_floor)
    if total > 0.0:
        probabilities = evidence / total
        nonzero = probabilities > 0.0
        entropy = -float(np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))
        normalized_entropy = entropy / np.log(len(probabilities))
    else:
        normalized_entropy = 0.0

    detected = (
        total >= config.minimum_total_evidence_dn
        and peak_snr >= config.noise_sigma_multiplier
    )
    saturated = (
        saturation_fraction_ge_250 > config.maximum_saturation_fraction_ge_250
        or saturation_fraction_eq_255 > config.maximum_saturation_fraction_eq_255
    )
    localized = (
        detected
        and peak_to_background >= config.minimum_peak_to_background_ratio
        and normalized_entropy <= config.maximum_normalized_entropy
        and not saturated
    )
    position_mm: float | None = None
    normalized_position: float | None = None
    if localized:
        coordinates = np.linspace(0.0, 1.0, len(evidence), dtype=np.float64)
        normalized_position = float(np.dot(coordinates, evidence) / total)
        try:
            position_mm = canonical_position_to_mm(
                normalized_position,
                led_coordinates,
                maximum_extrapolation_mm=config.maximum_extrapolation_mm,
            )
        except ValueError:
            localized = False
            normalized_position = None

    if saturated:
        status = "invalid: saturated selected pixels"
    elif detected and peak_to_background < config.minimum_peak_to_background_ratio:
        status = "invalid: response lacks spatial contrast"
    elif detected and normalized_entropy > config.maximum_normalized_entropy:
        status = "invalid: response is spatially diffuse"
    elif detected and not localized:
        status = "invalid: contact centroid lies outside physical range"
    elif not detected:
        status = "no contact: insufficient optical evidence"
    else:
        status = "contact localized"
    return ContactLocalizationResult(
        valid=bool(not saturated and (localized or not detected)),
        status=status,
        contact_detected=bool(detected),
        position_mm=position_mm,
        normalized_position=normalized_position,
        total_evidence_dn=total,
        peak_snr=peak_snr,
        peak_to_background_ratio=float(peak_to_background),
        normalized_spatial_entropy=float(normalized_entropy),
        temporal_consistency_dn=float(temporal_consistency_dn),
        saturation_fraction_ge_250=float(saturation_fraction_ge_250),
        saturation_fraction_eq_255=float(saturation_fraction_eq_255),
        response_profile=response,
        evidence_profile=evidence,
    )


class OnlineContactLocalizer:
    """Minimal stateful orchestration around the pure geometry and optical steps.

    The acquisition layer must disable auto exposure and auto white balance and
    hold exposure, gain, and white balance fixed. This class owns no camera,
    GUI, filesystem, or hardware resources.
    """

    def __init__(
        self,
        config: ContactLocalizationConfig = ContactLocalizationConfig(),
        canonical_config: CanonicalFingerConfig = CanonicalFingerConfig(),
    ) -> None:
        self.config = config
        self.canonical_config = canonical_config
        self.reference_rgb: np.ndarray | None = None
        self.reference_fingertip: FingertipRegion | None = None
        self.reference_leds: LedGeometry | None = None
        self.reference_map: CanonicalFingerMap | None = None
        self.led_coordinates: np.ndarray | None = None
        self.previous_rgb: np.ndarray | None = None
        self.current_leds: LedGeometry | None = None
        self.unloaded_reference: UnloadedOpticalReference | None = None
        self._response_history: deque[np.ndarray] = deque(
            maxlen=config.temporal_median_window
        )

    @property
    def state(self) -> str:
        if self.reference_rgb is None:
            return "UNINITIALIZED"
        if self.unloaded_reference is None:
            return "GEOMETRY_READY"
        return "READY"

    def initialize_geometry(self, rgb: np.ndarray) -> LedGeometry:
        """Segment one unloaded image and initialize the physical LED anchors."""

        image = np.asarray(rgb)
        fingertip = segment_fingertip(image)
        leds = detect_leds(image, fingertip)
        canonical_map = build_canonical_map(fingertip, self.canonical_config)
        led_coordinates = landmark_longitudinal_coordinates(
            canonical_map,
            leds.landmarks_xy_px,
        )
        self.reference_rgb = image.copy()
        self.reference_fingertip = fingertip
        self.reference_leds = leds
        self.reference_map = canonical_map
        self.led_coordinates = led_coordinates
        self.previous_rgb = image.copy()
        self.current_leds = leds
        self.unloaded_reference = None
        self._response_history.clear()
        return leds

    def _registered_canonical(self, rgb: np.ndarray, leds: LedGeometry) -> np.ndarray:
        assert self.reference_leds is not None
        assert self.reference_map is not None
        transform = estimate_led_similarity_transform(self.reference_leds, leds)
        current_map = transform_canonical_map(self.reference_map, transform)
        return warp_to_canonical(rgb, current_map)

    def acquire_unloaded_baseline(
        self,
        rgb_frames: Iterable[np.ndarray],
    ) -> UnloadedOpticalReference:
        """Register and store a fixed multi-frame unloaded optical reference."""

        if self.state == "UNINITIALIZED":
            raise RuntimeError("initialize_geometry must be called first")
        assert self.previous_rgb is not None
        assert self.current_leds is not None
        previous = self.previous_rgb
        leds = self.current_leds
        canonical_frames = []
        for frame in rgb_frames:
            image = np.asarray(frame)
            leds = track_leds(previous, image, leds)
            canonical_frames.append(self._registered_canonical(image, leds))
            previous = image.copy()
        reference = build_unloaded_reference(canonical_frames, self.config)
        self.previous_rgb = previous
        self.current_leds = leds
        self.unloaded_reference = reference
        self._response_history.clear()
        return reference

    def _invalid_tracking_result(self, status: str) -> ContactLocalizationResult:
        assert self.reference_map is not None
        zeros = np.zeros(self.reference_map.shape[0], dtype=np.float64)
        return ContactLocalizationResult(
            valid=False,
            status=status,
            contact_detected=False,
            position_mm=None,
            normalized_position=None,
            total_evidence_dn=0.0,
            peak_snr=0.0,
            peak_to_background_ratio=0.0,
            normalized_spatial_entropy=0.0,
            temporal_consistency_dn=0.0,
            saturation_fraction_ge_250=0.0,
            saturation_fraction_eq_255=0.0,
            response_profile=zeros,
            evidence_profile=zeros,
        )

    def process(self, rgb: np.ndarray) -> ContactLocalizationResult:
        """Track, register, and localize one frame without adapting the baseline."""

        if self.state != "READY":
            raise RuntimeError("geometry and unloaded baseline must be initialized")
        assert self.previous_rgb is not None
        assert self.current_leds is not None
        assert self.reference_leds is not None
        assert self.reference_map is not None
        assert self.unloaded_reference is not None
        assert self.led_coordinates is not None
        image = np.asarray(rgb)
        try:
            tracked = track_leds(self.previous_rgb, image, self.current_leds)
        except (RuntimeError, ValueError) as tracking_error:
            try:
                recovered_fingertip = segment_fingertip(image)
                recovered_leds = reanchor_leds(
                    image,
                    recovered_fingertip,
                    self.current_leds,
                )
                estimate_led_similarity_transform(self.reference_leds, recovered_leds)
            except (RuntimeError, ValueError):
                return self._invalid_tracking_result(
                    f"invalid: LED tracking and absolute recovery failed ({tracking_error})"
                )
            self.previous_rgb = image.copy()
            self.current_leds = recovered_leds
            self._response_history.clear()
            return self._invalid_tracking_result(
                "invalid: LED tracking failed; geometry recovered for next frame"
            )

        canonical = self._registered_canonical(image, tracked)
        response, saturated_250, saturated_255 = compute_longitudinal_response(
            canonical,
            self.unloaded_reference.canonical_rgb,
            brightest_fraction=self.config.brightest_fraction,
            smoothing_sigma=self.config.longitudinal_smoothing_sigma,
        )
        self._response_history.append(response)
        filtered = causal_median_response(
            self._response_history,
            self.config.temporal_median_window,
        )
        temporal_consistency = float(np.sqrt(np.mean((response - filtered) ** 2)))
        result = localize_response_profile(
            filtered,
            self.unloaded_reference,
            self.led_coordinates,
            self.config,
            temporal_consistency_dn=temporal_consistency,
            saturation_fraction_ge_250=saturated_250,
            saturation_fraction_eq_255=saturated_255,
        )
        self.previous_rgb = image.copy()
        self.current_leds = tracked
        return result


__all__ = [
    "ContactLocalizationConfig",
    "ContactLocalizationResult",
    "OnlineContactLocalizer",
    "UnloadedOpticalReference",
    "build_unloaded_reference",
    "canonical_position_to_mm",
    "causal_median_response",
    "compute_longitudinal_response",
    "localize_response_profile",
]

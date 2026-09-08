"""Learning-free optical conditioning of calibrated torque-to-force models."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from experiments.localization import (
    CanonicalFingerConfig,
    CanonicalFingerMap,
    DenseProfileConfig,
    build_canonical_finger_map,
    detect_fingertip_boundary,
    extract_positive_response_profile,
    warp_to_canonical,
)


def _finite_tuple(name: str, values: tuple[float, ...]) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if not parsed or not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{name} must be a nonempty finite sequence")
    return parsed


@dataclass(frozen=True)
class LocationConditionedForceCalibration:
    """Supplied affine torque-to-force models, one per optical region."""

    region_locations_mm: tuple[float, ...]
    slopes_n_per_nm: tuple[float, ...]
    intercepts_n: tuple[float, ...]

    def __post_init__(self) -> None:
        locations = _finite_tuple("region_locations_mm", self.region_locations_mm)
        slopes = _finite_tuple("slopes_n_per_nm", self.slopes_n_per_nm)
        intercepts = _finite_tuple("intercepts_n", self.intercepts_n)
        if len(locations) < 2:
            raise ValueError("calibration requires at least two optical regions")
        if len(slopes) != len(locations) or len(intercepts) != len(locations):
            raise ValueError("calibration arrays must have equal lengths")
        if any(right <= left for left, right in zip(locations, locations[1:])):
            raise ValueError("region_locations_mm must be strictly increasing")
        object.__setattr__(self, "region_locations_mm", locations)
        object.__setattr__(self, "slopes_n_per_nm", slopes)
        object.__setattr__(self, "intercepts_n", intercepts)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> LocationConditionedForceCalibration:
        """Construct calibration from a JSON-compatible mapping."""

        return cls(
            region_locations_mm=tuple(values["region_locations_mm"]),
            slopes_n_per_nm=tuple(values["slopes_n_per_nm"]),
            intercepts_n=tuple(values["intercepts_n"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> LocationConditionedForceCalibration:
        """Load supplied coefficients without fitting or modifying them."""

        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("force calibration JSON must contain an object")
        return cls.from_mapping(values)

    def to_mapping(self) -> dict[str, list[float]]:
        """Return a minimal JSON-compatible representation."""

        return {
            "region_locations_mm": list(self.region_locations_mm),
            "slopes_n_per_nm": list(self.slopes_n_per_nm),
            "intercepts_n": list(self.intercepts_n),
        }

    def predict_regions(self, relative_torque_nm: float) -> tuple[float, ...]:
        """Evaluate every fixed affine model at bias-corrected torque."""

        torque = float(relative_torque_nm)
        if not math.isfinite(torque):
            raise ValueError("relative_torque_nm must be finite")
        return tuple(
            slope * torque + intercept
            for slope, intercept in zip(
                self.slopes_n_per_nm,
                self.intercepts_n,
                strict=True,
            )
        )


@dataclass(frozen=True)
class OnlineOpticalState:
    """Immutable result of one timestamped optical update."""

    timestamp_ns: int
    valid: bool
    status: str
    region_response: tuple[float, ...] | None
    region_weights: tuple[float, ...] | None
    contact_location_mm: float | None
    confidence: float | None
    processing_time_ms: float


@dataclass(frozen=True)
class OnlineForceEstimate:
    """Immutable result of one timestamped motor-torque update."""

    timestamp_ns: int
    valid: bool
    contact: bool
    estimated_force_n: float | None
    torque_nm: float
    torque_bias_nm: float | None
    contact_location_mm: float | None
    optical_weights: tuple[float, ...] | None
    optical_timestamp_ns: int | None
    optical_age_ms: float | None
    status: str
    processing_time_ms: float


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("rgb must be an H x W x 3 uint8 array")
    return image


def _region_basis(profile_length: int, region_count: int) -> np.ndarray:
    """Triangular finger-relative basis associated with an ordered LED array."""

    if profile_length < 2 or region_count < 2:
        raise ValueError("profile and region counts must each be at least two")
    samples = np.linspace(0.0, 1.0, profile_length, dtype=np.float64)
    spacing = 1.0 / region_count
    centers = (np.arange(region_count, dtype=np.float64) + 0.5) * spacing
    return np.maximum(
        1.0 - np.abs(samples[:, None] - centers[None, :]) / spacing,
        0.0,
    )


def _causal_median(responses: tuple[np.ndarray, ...]) -> np.ndarray:
    if not responses:
        raise ValueError("responses must be nonempty")
    return np.median(np.stack(responses), axis=0)


def _profile_to_region_response(
    profile: np.ndarray,
    region_count: int,
) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("profile must be a finite one-dimensional array")
    basis = _region_basis(len(values), region_count)
    support = np.sum(basis, axis=0)
    return np.sum(values[:, None] * basis, axis=0) / support


def _response_weights(
    response: np.ndarray,
    noise_threshold: np.ndarray,
    minimum_evidence_dn: float,
) -> tuple[np.ndarray | None, float | None]:
    values = np.asarray(response, dtype=np.float64)
    noise = np.asarray(noise_threshold, dtype=np.float64)
    if values.ndim != 1 or noise.shape != values.shape:
        raise ValueError("response and noise_threshold must be equal vectors")
    evidence = np.maximum(values - noise, 0.0)
    total = float(np.sum(evidence))
    if not math.isfinite(total) or total < minimum_evidence_dn:
        return None, None
    weights = evidence / total
    return weights, float(np.max(weights))


class CanonicalOpticalObserver:
    """Convert explicitly initialized finger images into spatial optical weights.

    Geometry is frozen between explicit initialization requests.  Repeating
    geometry initialization rebuilds only the image-to-canonical map; an
    existing canonical unloaded reference remains frozen.
    """

    def __init__(
        self,
        region_locations_mm: tuple[float, ...],
        *,
        geometry_frame_count: int = 30,
        baseline_frame_count: int = 30,
        temporal_window: int = 3,
        brightest_fraction: float = 0.10,
        longitudinal_smoothing_sigma_px: float = 2.0,
        baseline_noise_sigma: float = 4.0,
        minimum_evidence_dn: float = 1.0,
    ) -> None:
        self.region_locations_mm = _finite_tuple(
            "region_locations_mm", region_locations_mm
        )
        if len(self.region_locations_mm) < 2 or any(
            right <= left
            for left, right in zip(
                self.region_locations_mm,
                self.region_locations_mm[1:],
            )
        ):
            raise ValueError(
                "region_locations_mm must contain at least two increasing values"
            )
        for name, value in (
            ("geometry_frame_count", geometry_frame_count),
            ("baseline_frame_count", baseline_frame_count),
            ("temporal_window", temporal_window),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 < brightest_fraction <= 1.0:
            raise ValueError("brightest_fraction must be in (0, 1]")
        for name, value in (
            ("longitudinal_smoothing_sigma_px", longitudinal_smoothing_sigma_px),
            ("baseline_noise_sigma", baseline_noise_sigma),
            ("minimum_evidence_dn", minimum_evidence_dn),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if float(minimum_evidence_dn) == 0.0:
            raise ValueError("minimum_evidence_dn must be positive")

        self.geometry_frame_count = geometry_frame_count
        self.baseline_frame_count = baseline_frame_count
        self.temporal_window = temporal_window
        self.baseline_noise_sigma = float(baseline_noise_sigma)
        self.minimum_evidence_dn = float(minimum_evidence_dn)
        self._profile_config = DenseProfileConfig(
            mode="top10_red",
            transverse_reduction="top_fraction",
            top_fraction=float(brightest_fraction),
            longitudinal_smoothing_sigma_px=float(
                longitudinal_smoothing_sigma_px
            ),
        )
        self._canonical_config = CanonicalFingerConfig()
        self._lock = threading.Lock()
        self._generation = 0
        self._mode = "idle"
        self._geometry_frames: list[np.ndarray] = []
        self._baseline_frames: list[np.ndarray] = []
        self._canonical_map: CanonicalFingerMap | None = None
        self._source_shape: tuple[int, int] | None = None
        self._unloaded_reference: np.ndarray | None = None
        self._noise_threshold: np.ndarray | None = None
        self._response_history: deque[np.ndarray] = deque(maxlen=temporal_window)

    @property
    def geometry_ready(self) -> bool:
        with self._lock:
            return self._canonical_map is not None and self._mode not in {
                "geometry",
                "geometry_processing",
            }

    @property
    def baseline_ready(self) -> bool:
        with self._lock:
            return (
                self._unloaded_reference is not None
                and self._noise_threshold is not None
                and self._mode
                not in {
                    "geometry",
                    "geometry_processing",
                    "baseline",
                    "baseline_processing",
                }
            )

    def begin_geometry_initialization(self) -> None:
        """Explicitly collect unloaded frames for canonical geometry."""

        with self._lock:
            self._generation += 1
            self._mode = "geometry"
            self._geometry_frames.clear()
            self._response_history.clear()

    def begin_unloaded_baseline(self) -> None:
        """Explicitly collect canonical unloaded frames without later adaptation."""

        with self._lock:
            if self._canonical_map is None or self._mode == "geometry":
                raise RuntimeError("geometry must be ready before baseline acquisition")
            self._generation += 1
            self._mode = "baseline"
            self._baseline_frames.clear()
            self._response_history.clear()

    def update(self, timestamp_ns: int, rgb: np.ndarray) -> OnlineOpticalState:
        """Process one frame without holding a lock during image operations."""

        timestamp = _timestamp(timestamp_ns)
        image = _validate_rgb(rgb)
        started = perf_counter()
        with self._lock:
            mode = self._mode
            generation = self._generation
            canonical_map = self._canonical_map
            source_shape = self._source_shape
            unloaded = self._unloaded_reference
            noise = None if self._noise_threshold is None else self._noise_threshold.copy()

            if mode == "geometry":
                self._geometry_frames.append(image.copy())
                count = len(self._geometry_frames)
                if count < self.geometry_frame_count:
                    return self._invalid(
                        timestamp,
                        f"collecting geometry {count}/{self.geometry_frame_count}",
                        started,
                    )
                frames = tuple(self._geometry_frames)
                self._geometry_frames.clear()
                self._mode = "geometry_processing"
            else:
                frames = ()

        if mode == "geometry":
            try:
                median_rgb = np.median(np.stack(frames), axis=0).astype(np.uint8)
                region = detect_fingertip_boundary(median_rgb)
                new_map = build_canonical_finger_map(region, self._canonical_config)
            except Exception as error:
                with self._lock:
                    if generation == self._generation:
                        self._mode = "idle"
                return self._invalid(
                    timestamp,
                    f"geometry initialization failed: {type(error).__name__}: {error}",
                    started,
                )
            with self._lock:
                if generation != self._generation:
                    return self._invalid(
                        timestamp, "geometry initialization superseded", started
                    )
                self._canonical_map = new_map
                self._source_shape = image.shape[:2]
                self._mode = "idle"
                self._response_history.clear()
                baseline_ready = self._unloaded_reference is not None
            return self._invalid(
                timestamp,
                "geometry ready" if baseline_ready else "geometry ready; acquire baseline",
                started,
            )

        if canonical_map is None or source_shape is None:
            return self._invalid(timestamp, "geometry not initialized", started)
        if image.shape[:2] != source_shape:
            return self._invalid(timestamp, "RGB shape differs from geometry", started)

        canonical = warp_to_canonical(image, canonical_map)
        if mode == "baseline":
            with self._lock:
                if generation != self._generation or self._mode != "baseline":
                    return self._invalid(timestamp, "baseline acquisition superseded", started)
                self._baseline_frames.append(canonical.copy())
                count = len(self._baseline_frames)
                if count < self.baseline_frame_count:
                    return self._invalid(
                        timestamp,
                        f"collecting baseline {count}/{self.baseline_frame_count}",
                        started,
                    )
                baseline_frames = tuple(self._baseline_frames)
                self._baseline_frames.clear()
                self._mode = "baseline_processing"
            reference = np.median(np.stack(baseline_frames), axis=0).astype(np.uint8)
            baseline_responses = np.stack(
                [self._region_response(frame, reference) for frame in baseline_frames]
            )
            center = np.median(baseline_responses, axis=0)
            robust_sigma = 1.4826 * np.median(
                np.abs(baseline_responses - center[None, :]), axis=0
            )
            threshold = center + self.baseline_noise_sigma * robust_sigma
            with self._lock:
                if generation != self._generation:
                    return self._invalid(timestamp, "baseline acquisition superseded", started)
                self._unloaded_reference = reference
                self._noise_threshold = threshold
                self._mode = "idle"
                self._response_history.clear()
            return self._invalid(timestamp, "unloaded baseline ready", started)

        if mode in {"geometry_processing", "baseline_processing"}:
            return self._invalid(timestamp, f"{mode.replace('_', ' ')}", started)
        if unloaded is None or noise is None:
            return self._invalid(timestamp, "unloaded baseline not acquired", started)

        response = self._region_response(canonical, unloaded)
        with self._lock:
            if generation != self._generation:
                return self._invalid(timestamp, "optical update superseded", started)
            self._response_history.append(response)
            filtered = _causal_median(tuple(self._response_history))
        weights, confidence = _response_weights(
            filtered,
            noise,
            self.minimum_evidence_dn,
        )
        region_response = tuple(float(value) for value in filtered)
        if weights is None:
            return OnlineOpticalState(
                timestamp_ns=timestamp,
                valid=False,
                status="insufficient optical evidence",
                region_response=region_response,
                region_weights=None,
                contact_location_mm=None,
                confidence=None,
                processing_time_ms=(perf_counter() - started) * 1000.0,
            )
        location = float(np.dot(weights, np.asarray(self.region_locations_mm)))
        return OnlineOpticalState(
            timestamp_ns=timestamp,
            valid=True,
            status="optical weights ready",
            region_response=region_response,
            region_weights=tuple(float(value) for value in weights),
            contact_location_mm=location,
            confidence=confidence,
            processing_time_ms=(perf_counter() - started) * 1000.0,
        )

    def _region_response(
        self,
        canonical: np.ndarray,
        unloaded: np.ndarray,
    ) -> np.ndarray:
        profile = extract_positive_response_profile(
            canonical,
            unloaded,
            self._profile_config,
        )
        return _profile_to_region_response(profile, len(self.region_locations_mm))

    @staticmethod
    def _invalid(
        timestamp_ns: int,
        status: str,
        started: float,
    ) -> OnlineOpticalState:
        return OnlineOpticalState(
            timestamp_ns=timestamp_ns,
            valid=False,
            status=status,
            region_response=None,
            region_weights=None,
            contact_location_mm=None,
            confidence=None,
            processing_time_ms=(perf_counter() - started) * 1000.0,
        )


def _timestamp(timestamp_ns: int) -> int:
    if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool):
        raise TypeError("timestamp_ns must be an integer")
    if timestamp_ns < 0:
        raise ValueError("timestamp_ns must be nonnegative")
    return timestamp_ns


class OnlineForceEstimator:
    """Thread-safe asynchronous fusion of optical state and motor torque."""

    def __init__(
        self,
        calibration: LocationConditionedForceCalibration,
        *,
        contact_enter_threshold_nm: float,
        contact_exit_threshold_nm: float,
        optical_to_motor_offset_ns: int = 0,
        maximum_optical_age_ms: float = 100.0,
        optical_history_size: int = 16,
        geometry_frame_count: int = 30,
        baseline_frame_count: int = 30,
        temporal_window: int = 3,
        brightest_fraction: float = 0.10,
        longitudinal_smoothing_sigma_px: float = 2.0,
        baseline_noise_sigma: float = 4.0,
        minimum_evidence_dn: float = 1.0,
    ) -> None:
        if not isinstance(calibration, LocationConditionedForceCalibration):
            raise TypeError("calibration must be LocationConditionedForceCalibration")
        enter = float(contact_enter_threshold_nm)
        exit_ = float(contact_exit_threshold_nm)
        if not math.isfinite(enter) or not math.isfinite(exit_) or not enter > exit_ >= 0.0:
            raise ValueError("contact thresholds must satisfy enter > exit >= 0")
        if not isinstance(optical_to_motor_offset_ns, int) or isinstance(
            optical_to_motor_offset_ns, bool
        ):
            raise TypeError("optical_to_motor_offset_ns must be an integer")
        maximum_age = float(maximum_optical_age_ms)
        if not math.isfinite(maximum_age) or maximum_age < 0.0:
            raise ValueError("maximum_optical_age_ms must be finite and nonnegative")
        if (
            not isinstance(optical_history_size, int)
            or isinstance(optical_history_size, bool)
            or optical_history_size < 1
        ):
            raise ValueError("optical_history_size must be a positive integer")

        self.calibration = calibration
        self.contact_enter_threshold_nm = enter
        self.contact_exit_threshold_nm = exit_
        self.optical_to_motor_offset_ns = optical_to_motor_offset_ns
        self.maximum_optical_age_ns = round(maximum_age * 1.0e6)
        self._observer = CanonicalOpticalObserver(
            calibration.region_locations_mm,
            geometry_frame_count=geometry_frame_count,
            baseline_frame_count=baseline_frame_count,
            temporal_window=temporal_window,
            brightest_fraction=brightest_fraction,
            longitudinal_smoothing_sigma_px=longitudinal_smoothing_sigma_px,
            baseline_noise_sigma=baseline_noise_sigma,
            minimum_evidence_dn=minimum_evidence_dn,
        )
        self._lock = threading.Lock()
        self._optical_history: deque[OnlineOpticalState] = deque(
            maxlen=optical_history_size
        )
        self._torque_bias_nm: float | None = None
        self._contact = False

    @property
    def torque_bias_nm(self) -> float | None:
        with self._lock:
            return self._torque_bias_nm

    @property
    def geometry_ready(self) -> bool:
        return self._observer.geometry_ready

    @property
    def baseline_ready(self) -> bool:
        return self._observer.baseline_ready

    def begin_geometry_initialization(self) -> None:
        """Begin explicit unloaded geometry collection."""

        self._observer.begin_geometry_initialization()
        with self._lock:
            self._optical_history.clear()

    def begin_unloaded_baseline(self) -> None:
        """Begin explicit unloaded canonical reference collection."""

        self._observer.begin_unloaded_baseline()
        with self._lock:
            self._optical_history.clear()

    def set_torque_bias(self, torque_nm: float) -> None:
        """Set the explicitly acquired unloaded motor-torque bias."""

        value = float(torque_nm)
        if not math.isfinite(value):
            raise ValueError("torque bias must be finite")
        with self._lock:
            self._torque_bias_nm = value
            self._contact = False

    def update_optical(self, timestamp_ns: int, rgb: np.ndarray) -> OnlineOpticalState:
        """Publish one camera-derived state for later nonblocking motor fusion."""

        state = self._observer.update(timestamp_ns, rgb)
        if state.valid:
            with self._lock:
                ordered = list(self._optical_history)
                ordered.append(state)
                ordered.sort(key=lambda item: item.timestamp_ns)
                self._optical_history.clear()
                self._optical_history.extend(ordered[-self._optical_history.maxlen :])
        return state

    def update_torque(
        self,
        timestamp_ns: int,
        torque_nm: float,
    ) -> OnlineForceEstimate:
        """Fuse one motor sample with the latest nonfuture valid optical state."""

        timestamp = _timestamp(timestamp_ns)
        torque = float(torque_nm)
        if not math.isfinite(torque):
            raise ValueError("torque_nm must be finite")
        started = perf_counter()
        geometry_ready = self._observer.geometry_ready
        baseline_ready = self._observer.baseline_ready
        with self._lock:
            bias = self._torque_bias_nm
            if bias is not None:
                magnitude = abs(torque - bias)
                if self._contact:
                    if magnitude <= self.contact_exit_threshold_nm:
                        self._contact = False
                elif magnitude >= self.contact_enter_threshold_nm:
                    self._contact = True
            contact = self._contact
            history = tuple(self._optical_history)

        if bias is None:
            return self._invalid_force(timestamp, torque, None, contact, "torque bias not set", started)
        if not geometry_ready:
            return self._invalid_force(timestamp, torque, bias, contact, "geometry not initialized", started)
        if not baseline_ready:
            return self._invalid_force(timestamp, torque, bias, contact, "unloaded baseline not acquired", started)
        if not contact:
            optical = self._matched_optical(timestamp, history)
            optical_age_ms = None
            if optical is not None:
                optical_age_ms = (
                    timestamp
                    - optical.timestamp_ns
                    - self.optical_to_motor_offset_ns
                ) / 1.0e6
            return OnlineForceEstimate(
                timestamp_ns=timestamp,
                valid=True,
                contact=False,
                estimated_force_n=0.0,
                torque_nm=torque,
                torque_bias_nm=bias,
                contact_location_mm=(
                    None if optical is None else optical.contact_location_mm
                ),
                optical_weights=None if optical is None else optical.region_weights,
                optical_timestamp_ns=(
                    None if optical is None else optical.timestamp_ns
                ),
                optical_age_ms=optical_age_ms,
                status="no contact",
                processing_time_ms=(perf_counter() - started) * 1000.0,
            )

        optical = self._matched_optical(timestamp, history)
        if optical is None:
            return self._invalid_force(
                timestamp, torque, bias, True, "no nonfuture optical state", started
            )
        aligned_timestamp = optical.timestamp_ns + self.optical_to_motor_offset_ns
        age_ns = timestamp - aligned_timestamp
        age_ms = age_ns / 1.0e6
        if age_ns > self.maximum_optical_age_ns:
            return OnlineForceEstimate(
                timestamp_ns=timestamp,
                valid=False,
                contact=True,
                estimated_force_n=None,
                torque_nm=torque,
                torque_bias_nm=bias,
                contact_location_mm=optical.contact_location_mm,
                optical_weights=optical.region_weights,
                optical_timestamp_ns=optical.timestamp_ns,
                optical_age_ms=age_ms,
                status="optical state stale",
                processing_time_ms=(perf_counter() - started) * 1000.0,
            )
        assert optical.region_weights is not None
        region_forces = self.calibration.predict_regions(torque - bias)
        estimated = float(np.dot(optical.region_weights, region_forces))
        return OnlineForceEstimate(
            timestamp_ns=timestamp,
            valid=True,
            contact=True,
            estimated_force_n=estimated,
            torque_nm=torque,
            torque_bias_nm=bias,
            contact_location_mm=optical.contact_location_mm,
            optical_weights=optical.region_weights,
            optical_timestamp_ns=optical.timestamp_ns,
            optical_age_ms=age_ms,
            status="force estimate ready",
            processing_time_ms=(perf_counter() - started) * 1000.0,
        )

    def _matched_optical(
        self,
        motor_timestamp_ns: int,
        history: tuple[OnlineOpticalState, ...],
    ) -> OnlineOpticalState | None:
        eligible = [
            state
            for state in history
            if state.timestamp_ns + self.optical_to_motor_offset_ns
            <= motor_timestamp_ns
        ]
        if not eligible:
            return None
        return max(eligible, key=lambda state: state.timestamp_ns)

    @staticmethod
    def _invalid_force(
        timestamp_ns: int,
        torque_nm: float,
        torque_bias_nm: float | None,
        contact: bool,
        status: str,
        started: float,
    ) -> OnlineForceEstimate:
        return OnlineForceEstimate(
            timestamp_ns=timestamp_ns,
            valid=False,
            contact=contact,
            estimated_force_n=None,
            torque_nm=torque_nm,
            torque_bias_nm=torque_bias_nm,
            contact_location_mm=None,
            optical_weights=None,
            optical_timestamp_ns=None,
            optical_age_ms=None,
            status=status,
            processing_time_ms=(perf_counter() - started) * 1000.0,
        )


__all__ = [
    "CanonicalOpticalObserver",
    "LocationConditionedForceCalibration",
    "OnlineForceEstimate",
    "OnlineForceEstimator",
    "OnlineOpticalState",
]

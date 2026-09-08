"""Stateful online five-LED contact tracking without camera or GUI ownership."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading

import numpy as np

from lumo.fingertip import LED_CENTERS_Y_MM

from .contact import (
    FEATURE_NOISE_FLOOR_DN,
    LedArrayGeometry,
    brightest_red_features,
    contact_image_point,
    detect_led_array,
    estimate_contact_position,
    reanchor_led_array,
    track_led_array,
    unloaded_baseline_statistics,
)
from .fingertip_boundary import detect_fingertip_boundary


@dataclass(frozen=True)
class LiveLedContactResult:
    """One frame's currently available online contact result."""

    status: str
    geometry_ready: bool
    baseline_ready: bool
    contact_detected: bool | None
    contact_location_mm: float | None
    contact_score_z: float | None
    top_two_margin_dn: float | None
    optical_response_dn: tuple[float, ...] | None
    landmarks_xy_px: tuple[tuple[float, float], ...] | None
    contact_point_xy_px: tuple[float, float] | None


class LiveLedContactTracker:
    """Reuse the production LED detector, rigid tracker, and unloaded contact gate."""

    def __init__(
        self,
        *,
        calibration_frame_count: int = 30,
        baseline_frame_count: int = 30,
        feature_median_window: int = 3,
        reanchor_frame_count: int = 30,
    ) -> None:
        for name, value in (
            ("calibration_frame_count", calibration_frame_count),
            ("baseline_frame_count", baseline_frame_count),
            ("feature_median_window", feature_median_window),
            ("reanchor_frame_count", reanchor_frame_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._calibration_frames: deque[np.ndarray] = deque(
            maxlen=calibration_frame_count
        )
        self._baseline_samples: deque[np.ndarray] = deque(
            maxlen=baseline_frame_count
        )
        self._feature_history: deque[np.ndarray] = deque(
            maxlen=feature_median_window
        )
        self._reanchor_frame_count = reanchor_frame_count
        self._recalibrate_requested = threading.Event()
        self._baseline_requested = threading.Event()
        self._geometry: LedArrayGeometry | None = None
        self._previous_rgb: np.ndarray | None = None
        self._baseline: np.ndarray | None = None
        self._noise_sigma: np.ndarray | None = None
        self._baseline_collecting = False
        self._no_contact_frames = 0
        self._calibration_error: str | None = None

    def request_recalibration(self) -> None:
        """Request geometry calibration in the camera worker thread."""

        self._recalibrate_requested.set()

    def request_unloaded_baseline(self) -> None:
        """Request a fresh unloaded baseline in the camera worker thread."""

        self._baseline_requested.set()

    def process(self, rgb: np.ndarray) -> LiveLedContactResult:
        """Process one owned RGB frame using the current online LED pipeline."""

        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("rgb must be an H x W x 3 uint8 image")
        if self._recalibrate_requested.is_set():
            self._recalibrate_requested.clear()
            self._reset_geometry()

        if self._geometry is not None and self._previous_rgb is not None:
            try:
                self._geometry = track_led_array(
                    self._previous_rgb,
                    image,
                    self._geometry,
                )
            except RuntimeError:
                self._reset_geometry()

        if self._geometry is None:
            self._previous_rgb = image
            if self._calibration_error is not None:
                return self._result(
                    f"geometry calibration failed: {self._calibration_error}"
                )
            self._calibration_frames.append(image)
            count = len(self._calibration_frames)
            if count < self._calibration_frames.maxlen:
                return self._result(
                    f"calibrating geometry {count}/{self._calibration_frames.maxlen}"
                )
            calibration_rgb = np.median(
                np.stack(self._calibration_frames), axis=0
            ).astype(np.uint8)
            try:
                boundary = detect_fingertip_boundary(calibration_rgb)
                self._geometry = detect_led_array(
                    calibration_rgb,
                    search_mask=boundary.search_mask,
                )
            except RuntimeError as error:
                self._calibration_error = str(error)
                return self._result(
                    f"geometry calibration failed: {self._calibration_error}"
                )
            finally:
                self._calibration_frames.clear()
            self._previous_rgb = image
            return self._result("geometry ready; acquire unloaded baseline")

        features = brightest_red_features(image, self._geometry)
        if self._baseline_requested.is_set():
            self._baseline_requested.clear()
            self._baseline = None
            self._noise_sigma = None
            self._baseline_collecting = True
            self._baseline_samples.clear()
            self._feature_history.clear()

        if self._baseline_collecting:
            self._baseline_samples.append(features)
            count = len(self._baseline_samples)
            if count == self._baseline_samples.maxlen:
                self._baseline, self._noise_sigma = unloaded_baseline_statistics(
                    np.stack(self._baseline_samples)
                )
                self._baseline_collecting = False
                self._baseline_samples.clear()
                self._feature_history.clear()
                status = "unloaded baseline ready"
            else:
                status = (
                    f"collecting unloaded baseline {count}/"
                    f"{self._baseline_samples.maxlen}"
                )
            self._previous_rgb = image
            return self._result(status)

        if self._baseline is None or self._noise_sigma is None:
            self._previous_rgb = image
            return self._result("geometry ready; acquire unloaded baseline")

        self._feature_history.append(features)
        filtered = np.median(np.stack(self._feature_history), axis=0)
        estimate = estimate_contact_position(
            filtered,
            self._baseline,
            self._noise_sigma,
            np.asarray(LED_CENTERS_Y_MM, dtype=np.float64),
        )
        standardized = np.maximum(estimate.response, 0.0) / np.maximum(
            self._noise_sigma,
            FEATURE_NOISE_FLOOR_DN,
        )
        score_z = float(np.max(standardized))
        point = contact_image_point(estimate, self._geometry)

        if estimate.contact_detected:
            self._no_contact_frames = 0
        else:
            self._no_contact_frames += 1
            if self._no_contact_frames >= self._reanchor_frame_count:
                try:
                    self._geometry = reanchor_led_array(image, self._geometry)
                    self._feature_history.clear()
                except RuntimeError:
                    pass
                self._no_contact_frames = 0

        self._previous_rgb = image
        return self._result(
            "contact" if estimate.contact_detected else "no contact",
            contact_detected=estimate.contact_detected,
            contact_location_mm=estimate.position_mm,
            contact_score_z=score_z,
            top_two_margin_dn=estimate.top_two_margin,
            optical_response_dn=tuple(float(value) for value in estimate.response),
            contact_point_xy_px=(
                None if point is None else (float(point[0]), float(point[1]))
            ),
        )

    def _reset_geometry(self) -> None:
        self._geometry = None
        self._previous_rgb = None
        self._baseline = None
        self._noise_sigma = None
        self._baseline_collecting = False
        self._no_contact_frames = 0
        self._calibration_error = None
        self._calibration_frames.clear()
        self._baseline_samples.clear()
        self._feature_history.clear()

    def _result(
        self,
        status: str,
        *,
        contact_detected: bool | None = None,
        contact_location_mm: float | None = None,
        contact_score_z: float | None = None,
        top_two_margin_dn: float | None = None,
        optical_response_dn: tuple[float, ...] | None = None,
        contact_point_xy_px: tuple[float, float] | None = None,
    ) -> LiveLedContactResult:
        landmarks = None
        if self._geometry is not None:
            landmarks = tuple(
                (float(point[0]), float(point[1]))
                for point in self._geometry.landmarks_xy_px
            )
        return LiveLedContactResult(
            status=status,
            geometry_ready=self._geometry is not None,
            baseline_ready=self._baseline is not None,
            contact_detected=contact_detected,
            contact_location_mm=contact_location_mm,
            contact_score_z=contact_score_z,
            top_two_margin_dn=top_two_margin_dn,
            optical_response_dn=optical_response_dn,
            landmarks_xy_px=landmarks,
            contact_point_xy_px=contact_point_xy_px,
        )


__all__ = ["LiveLedContactResult", "LiveLedContactTracker"]

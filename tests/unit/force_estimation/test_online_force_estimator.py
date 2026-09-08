from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import experiments.force_estimation.online_force_estimator as estimator_module
from experiments.force_estimation import (
    CanonicalOpticalObserver,
    LocationConditionedForceCalibration,
    OnlineForceEstimator,
    OnlineOpticalState,
)
from experiments.localization import FingertipBoundaryRegion


def _calibration() -> LocationConditionedForceCalibration:
    return LocationConditionedForceCalibration(
        region_locations_mm=(0.0, 10.0, 20.0, 30.0, 40.0),
        slopes_n_per_nm=(1.0, 2.0, 3.0, 4.0, 5.0),
        intercepts_n=(0.0, 0.0, 0.0, 0.0, 0.0),
    )


def _optical_state(
    timestamp_ns: int,
    weights: tuple[float, ...],
) -> OnlineOpticalState:
    locations = np.arange(len(weights), dtype=float) * 10.0
    return OnlineOpticalState(
        timestamp_ns=timestamp_ns,
        valid=True,
        status="synthetic",
        region_response=weights,
        region_weights=weights,
        contact_location_mm=float(np.dot(weights, locations)),
        confidence=max(weights),
        processing_time_ms=0.0,
    )


class _ReadyObserver:
    geometry_ready = True
    baseline_ready = True

    def __init__(self) -> None:
        self.next_state = _optical_state(0, (1.0, 0.0, 0.0, 0.0, 0.0))

    def begin_geometry_initialization(self) -> None:
        pass

    def begin_unloaded_baseline(self) -> None:
        pass

    def update(self, timestamp_ns: int, rgb: np.ndarray) -> OnlineOpticalState:
        del timestamp_ns, rgb
        return self.next_state


def _ready_estimator(
    *,
    maximum_optical_age_ms: float = 100.0,
    optical_to_motor_offset_ns: int = 0,
) -> tuple[OnlineForceEstimator, _ReadyObserver]:
    estimator = OnlineForceEstimator(
        _calibration(),
        contact_enter_threshold_nm=0.5,
        contact_exit_threshold_nm=0.2,
        maximum_optical_age_ms=maximum_optical_age_ms,
        optical_to_motor_offset_ns=optical_to_motor_offset_ns,
    )
    observer = _ReadyObserver()
    estimator._observer = observer  # type: ignore[assignment]
    estimator.set_torque_bias(1.0)
    return estimator, observer


def _region() -> FingertipBoundaryRegion:
    rows = np.arange(10, 90, dtype=np.float64)
    left = np.full_like(rows, 20.0)
    right = np.full_like(rows, 80.0)
    mask = np.zeros((100, 120), dtype=bool)
    mask[10:90, 20:81] = True
    return FingertipBoundaryRegion(
        dorsal_boundary_xy_px=np.column_stack((left, rows)),
        palmar_boundary_xy_px=np.column_stack((right, rows)),
        search_mask=mask,
        core_y_span=(10, 90),
        estimated_pad_width_px=60.0,
    )


def test_constructor_is_passive_and_baseline_is_explicit() -> None:
    estimator = OnlineForceEstimator(
        _calibration(),
        contact_enter_threshold_nm=0.5,
        contact_exit_threshold_nm=0.2,
    )

    assert not estimator.geometry_ready
    assert not estimator.baseline_ready
    assert not estimator.update_torque(1, 0.0).valid
    try:
        estimator.begin_unloaded_baseline()
    except RuntimeError as error:
        assert "geometry" in str(error)
    else:
        raise AssertionError("baseline acquisition must require explicit geometry")


def test_spatial_basis_produces_one_region_and_midpoint_blending() -> None:
    profile = np.zeros(1001)
    profile[500] = 1.0
    centered = estimator_module._profile_to_region_response(profile, 5)
    centered_weights, _ = estimator_module._response_weights(
        centered, np.zeros(5), 1.0e-12
    )
    assert centered_weights is not None
    assert np.argmax(centered_weights) == 2
    assert centered_weights[2] > 0.99

    halfway = np.zeros(1001)
    halfway[400] = 1.0
    blended = estimator_module._profile_to_region_response(halfway, 5)
    blended_weights, _ = estimator_module._response_weights(
        blended, np.zeros(5), 1.0e-12
    )
    assert blended_weights is not None
    assert np.all(blended_weights >= 0.0)
    assert np.isclose(np.sum(blended_weights), 1.0)
    assert np.allclose(blended_weights[1:3], (0.5, 0.5), atol=0.01)


def test_insufficient_signal_and_causal_median_outlier_rejection() -> None:
    weights, confidence = estimator_module._response_weights(
        np.ones(5), np.ones(5), 0.1
    )
    assert weights is None
    assert confidence is None

    normal = np.array((1.0, 2.0, 3.0))
    outlier = np.array((100.0, -100.0, 200.0))
    filtered = estimator_module._causal_median((normal, outlier, normal))
    assert np.array_equal(filtered, normal)


def test_geometry_and_baseline_reuse_production_canonicalization(monkeypatch) -> None:
    calls = {"detect": 0, "build": 0, "warp": 0}
    original_build = estimator_module.build_canonical_finger_map
    original_warp = estimator_module.warp_to_canonical

    def detect(rgb: np.ndarray) -> FingertipBoundaryRegion:
        assert rgb.shape == (100, 120, 3)
        calls["detect"] += 1
        return _region()

    def build(region, config):
        calls["build"] += 1
        return original_build(region, config)

    def warp(rgb, canonical_map):
        calls["warp"] += 1
        return original_warp(rgb, canonical_map)

    monkeypatch.setattr(estimator_module, "detect_fingertip_boundary", detect)
    monkeypatch.setattr(estimator_module, "build_canonical_finger_map", build)
    monkeypatch.setattr(estimator_module, "warp_to_canonical", warp)
    observer = CanonicalOpticalObserver(
        _calibration().region_locations_mm,
        geometry_frame_count=1,
        baseline_frame_count=1,
        temporal_window=1,
        baseline_noise_sigma=0.0,
        minimum_evidence_dn=0.1,
    )
    unloaded = np.full((100, 120, 3), 20, dtype=np.uint8)

    assert "not initialized" in observer.update(0, unloaded).status
    observer.begin_geometry_initialization()
    assert "geometry ready" in observer.update(1, unloaded).status
    assert not observer.update(2, unloaded).valid
    observer.begin_unloaded_baseline()
    assert "baseline ready" in observer.update(3, unloaded).status
    loaded = unloaded.copy()
    loaded[10:90, 20:81, 0] = 60
    result = observer.update(4, loaded)

    assert result.valid
    assert result.region_weights is not None
    assert np.isclose(sum(result.region_weights), 1.0)
    assert calls == {"detect": 1, "build": 1, "warp": 3}


def test_force_fusion_bias_zero_force_and_hysteresis() -> None:
    estimator, observer = _ready_estimator()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    observer.next_state = _optical_state(100, (0.0, 0.0, 1.0, 0.0, 0.0))
    estimator.update_optical(100, image)

    selected = estimator.update_torque(110, 2.0)
    assert selected.valid
    assert selected.contact
    assert selected.estimated_force_n == 3.0

    observer.next_state = _optical_state(120, (0.0, 0.5, 0.5, 0.0, 0.0))
    estimator.update_optical(120, image)
    blended = estimator.update_torque(130, 2.0)
    assert blended.estimated_force_n == 2.5

    assert estimator.update_torque(140, 1.3).contact
    released = estimator.update_torque(150, 1.2)
    assert released.valid
    assert not released.contact
    assert released.estimated_force_n == 0.0


def test_timestamp_matching_reuse_staleness_offset_and_no_future_sample() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    estimator, observer = _ready_estimator(maximum_optical_age_ms=5.0)
    observer.next_state = _optical_state(
        10_000_000, (1.0, 0.0, 0.0, 0.0, 0.0)
    )
    estimator.update_optical(10_000_000, image)
    assert estimator.update_torque(12_000_000, 2.0).valid
    reused = estimator.update_torque(14_000_000, 2.0)
    assert reused.valid
    assert reused.optical_age_ms == 4.0
    assert not estimator.update_torque(16_000_000, 2.0).valid

    future, future_observer = _ready_estimator()
    future_observer.next_state = _optical_state(
        20_000_000, (1.0, 0.0, 0.0, 0.0, 0.0)
    )
    future.update_optical(20_000_000, image)
    assert "nonfuture" in future.update_torque(19_000_000, 2.0).status

    offset, offset_observer = _ready_estimator(optical_to_motor_offset_ns=3_000_000)
    offset_observer.next_state = _optical_state(
        20_000_000, (1.0, 0.0, 0.0, 0.0, 0.0)
    )
    offset.update_optical(20_000_000, image)
    assert not offset.update_torque(22_000_000, 2.0).valid
    matched = offset.update_torque(24_000_000, 2.0)
    assert matched.valid
    assert matched.optical_age_ms == 1.0


def test_interleaved_optical_and_torque_updates_are_thread_safe() -> None:
    estimator, observer = _ready_estimator(maximum_optical_age_ms=1000.0)
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    def optical_updates() -> None:
        for index in range(100):
            observer.next_state = _optical_state(
                index * 1_000_000,
                (0.0, 0.0, 1.0, 0.0, 0.0),
            )
            estimator.update_optical(index * 1_000_000, image)

    def torque_updates() -> list[bool]:
        return [
            estimator.update_torque(index * 1_000_000, 2.0).contact
            for index in range(100)
        ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        optical_future = pool.submit(optical_updates)
        torque_future = pool.submit(torque_updates)
        optical_future.result()
        contacts = torque_future.result()

    assert all(contacts)

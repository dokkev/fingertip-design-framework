import numpy as np
import cv2

from algorithm.contact_localization import (
    ContactLocalizationConfig,
    OnlineContactLocalizer,
    build_unloaded_reference,
    canonical_position_to_mm,
    causal_median_response,
    localize_response_profile,
)


LED_COORDINATES = np.asarray([0.10, 0.30, 0.50, 0.70, 0.90])


def _config() -> ContactLocalizationConfig:
    return ContactLocalizationConfig(unloaded_frame_count=3)


def _reference():
    frames = [np.zeros((256, 32, 3), dtype=np.uint8) for _ in range(3)]
    return build_unloaded_reference(frames, _config())


def _gaussian(center: float, sigma: float = 0.018, amplitude: float = 18.0) -> np.ndarray:
    coordinates = np.linspace(0.0, 1.0, 256)
    return amplitude * np.exp(-0.5 * ((coordinates - center) / sigma) ** 2)


def _synthetic_rgb() -> np.ndarray:
    rgb = np.full((240, 240, 3), 10, dtype=np.uint8)
    cv2.rectangle(rgb, (70, 20), (150, 220), (20, 180, 190), -1)
    for y_coordinate in (55, 92, 128, 165, 202):
        cv2.circle(rgb, (100, y_coordinate), 5, (255, 220, 80), -1)
    return rgb


def test_zero_mad_uses_nonzero_absolute_noise_floor() -> None:
    reference = _reference()

    np.testing.assert_allclose(reference.response_sigma_dn, 0.75)
    np.testing.assert_allclose(reference.response_threshold_dn, 3.0)


def test_single_led_peak_maps_to_known_physical_position() -> None:
    result = localize_response_profile(_gaussian(0.30), _reference(), LED_COORDINATES, _config())

    assert result.valid and result.contact_detected
    assert result.position_mm is not None
    assert abs(result.position_mm - 11.0) < 0.5


def test_midpoint_peak_interpolates_continuously_between_leds() -> None:
    result = localize_response_profile(_gaussian(0.40), _reference(), LED_COORDINATES, _config())

    assert result.position_mm is not None
    assert abs(result.position_mm - 16.5) < 0.5


def test_distal_and_proximal_peaks_use_bounded_extrapolation() -> None:
    distal = localize_response_profile(_gaussian(0.05), _reference(), LED_COORDINATES, _config())
    proximal = localize_response_profile(_gaussian(0.95), _reference(), LED_COORDINATES, _config())

    assert distal.position_mm is not None and -3.5 < distal.position_mm < -2.0
    assert proximal.position_mm is not None and 46.0 < proximal.position_mm < 47.5


def test_broad_symmetric_response_has_centered_centroid() -> None:
    result = localize_response_profile(
        _gaussian(0.50, sigma=0.12, amplitude=18.0),
        _reference(),
        LED_COORDINATES,
        _config(),
    )

    assert result.valid and result.position_mm is not None
    assert abs(result.position_mm - 22.0) < 0.5


def test_uniform_brightness_drift_is_not_localized_as_center_contact() -> None:
    result = localize_response_profile(
        np.full(256, 12.0),
        _reference(),
        LED_COORDINATES,
        _config(),
    )

    assert result.contact_detected
    assert not result.valid
    assert result.position_mm is None
    assert "spatial" in result.status


def test_small_response_is_reported_as_no_contact() -> None:
    result = localize_response_profile(
        _gaussian(0.50, amplitude=2.0),
        _reference(),
        LED_COORDINATES,
        _config(),
    )

    assert result.valid
    assert not result.contact_detected
    assert result.position_mm is None


def test_causal_temporal_median_rejects_one_frame_outlier() -> None:
    quiet = np.zeros(16)
    outlier = np.zeros(16)
    outlier[8] = 100.0

    filtered = causal_median_response([quiet, quiet, outlier], window=3)

    np.testing.assert_array_equal(filtered, quiet)


def test_saturated_selected_pixels_invalidate_contact() -> None:
    result = localize_response_profile(
        _gaussian(0.50),
        _reference(),
        LED_COORDINATES,
        _config(),
        saturation_fraction_ge_250=0.30,
        saturation_fraction_eq_255=0.10,
    )

    assert result.contact_detected
    assert not result.valid
    assert result.position_mm is None
    assert "saturated" in result.status


def test_physical_mapping_uses_only_led_coordinates_and_pitch() -> None:
    assert canonical_position_to_mm(0.60, LED_COORDINATES) == 27.5


def test_online_wrapper_initializes_baseline_and_localizes_registered_frame() -> None:
    unloaded = _synthetic_rgb()
    localizer = OnlineContactLocalizer(_config())
    localizer.initialize_geometry(unloaded)
    assert localizer.state == "GEOMETRY_READY"
    localizer.acquire_unloaded_baseline([unloaded.copy() for _ in range(3)])
    assert localizer.state == "READY"
    loaded = unloaded.copy()
    for row in range(112, 145):
        increase = int(50 * np.exp(-0.5 * ((row - 128) / 7) ** 2))
        loaded[row, 110:145, 1] = np.clip(
            loaded[row, 110:145, 1].astype(int) + increase,
            0,
            255,
        )

    result = localizer.process(loaded)

    assert result.valid and result.contact_detected
    assert result.position_mm is not None
    assert abs(result.position_mm - 22.0) < 1.0

"""Focused Mitsuba source, sampler, and side-field metric tests."""

from __future__ import annotations

import numpy as np
import pytest

mi = pytest.importorskip("mitsuba")

from mesh import mesh_settings_for_level
from model import Fingertip, FingertipParameters
from optics.mitsuba import Camera, MitsubaRenderer, RenderSettings
from validation.optics.pre_bo_mitsuba_light_field import (
    _camera_for_union,
    _source_positions_mm,
    normalized_total_variation,
    scalar_light_field,
)


def _tip_and_mesh() -> tuple[Fingertip, object]:
    tip = Fingertip(FingertipParameters())
    return tip, tip.mesh(mesh_settings_for_level("medium"))


def _camera() -> Camera:
    return Camera(
        position_mm=(20.0, -2.0, 0.0),
        target_mm=(0.0, -2.0, 0.0),
        up=(0.0, 0.0, 1.0),
        resolution_px=(16, 16),
        orthographic_scale_mm=40.0,
    )


def test_five_source_positions_use_led_center_and_required_z_locations() -> None:
    tip, _ = _tip_and_mesh()
    positions = _source_positions_mm(tip)

    assert len(positions) == 5
    assert [position[0] for position in positions] == [0.0] * 5
    assert [position[1] for position in positions] == [-5.0] * 5
    assert [position[2] for position in positions] == [-19.9, -8.9, 2.1, 13.1, 24.1]


def test_single_cell_source_is_one_package_center_at_z_zero() -> None:
    tip, _ = _tip_and_mesh()
    positions = _source_positions_mm(tip, (0.0,))

    assert positions == ((0.0, -5.0, 0.0),)


def test_mitsuba_accepts_five_sources_and_updates_all_intensities() -> None:
    tip, mesh = _tip_and_mesh()
    renderer = MitsubaRenderer(
        tip,
        mesh,
        depth_mm=64.8,
        camera=_camera(),
        settings=RenderSettings(source_epsilon_mm=0.0),
        source_positions_mm=_source_positions_mm(tip),
    )

    session = renderer._session
    assert session._led_intensity_keys == tuple(
        f"led_{index}.intensity.value" for index in range(5)
    )
    session.set_led_relative_power(2.0)
    parameters = mi.traverse(session._scene)
    values = [parameters[key] for key in session._led_intensity_keys]
    assert all(np.allclose(values[0], value) for value in values[1:])


def test_single_source_path_remains_the_existing_led_path() -> None:
    tip, mesh = _tip_and_mesh()
    renderer = MitsubaRenderer(
        tip,
        mesh,
        camera=_camera(),
        settings=RenderSettings(),
    )

    assert renderer._session._led_intensity_keys == ("led.intensity.value",)


def test_zero_source_epsilon_does_not_weaken_optical_depth_validation() -> None:
    assert RenderSettings(source_epsilon_mm=0.0).source_epsilon_mm == 0.0
    with pytest.raises(ValueError, match="optical_depth_mm"):
        RenderSettings(optical_depth_mm=0.0)


def test_fixed_sampler_is_derived_once_from_both_morphologies() -> None:
    nominal_tip, nominal_mesh = _tip_and_mesh()
    candidate_tip = Fingertip(
        FingertipParameters(
            flat_pad_height=3.937175708822906,
            semielliptical_pad_height=7.309789158403873,
            stem_width=7.289858109783381,
            stem_height=5.102298432029784,
            void_width=0.6931721470318735,
            void_height=1.2690955214202404,
        )
    )
    candidate_mesh = candidate_tip.mesh(mesh_settings_for_level("medium"))

    camera, configuration = _camera_for_union(
        ((nominal_tip, nominal_mesh), (candidate_tip, candidate_mesh))
    )

    assert camera.resolution_px == (384, 1024)
    assert camera.up == (0.0, 0.0, 1.0)
    assert camera.orthographic_scale_mm >= 68.8
    assert configuration["union_z_bounds_mm"] == [-32.4, 32.4]


def test_single_cell_sampler_uses_the_11_mm_shared_extrusion() -> None:
    nominal_tip, nominal_mesh = _tip_and_mesh()
    candidate_tip = Fingertip(
        FingertipParameters(
            flat_pad_height=3.937175708822906,
            semielliptical_pad_height=7.309789158403873,
            stem_width=7.289858109783381,
            stem_height=5.102298432029784,
            void_width=0.6931721470318735,
            void_height=1.2690955214202404,
        )
    )
    candidate_mesh = candidate_tip.mesh(mesh_settings_for_level("medium"))

    _, configuration = _camera_for_union(
        ((nominal_tip, nominal_mesh), (candidate_tip, candidate_mesh)),
        extrusion_length_mm=11.0,
    )

    assert configuration["union_z_bounds_mm"] == [-5.5, 5.5]


def test_side_field_metrics_use_channel_sum_and_independent_energy_normalization() -> None:
    left_rgb = np.zeros((2, 2, 3), dtype=float)
    right_rgb = np.zeros((2, 2, 3), dtype=float)
    left_rgb[0, 0] = (1.0, 2.0, 3.0)
    right_rgb[1, 1] = (1.0, 2.0, 3.0)

    left = scalar_light_field(left_rgb)
    right = scalar_light_field(right_rgb)

    assert left[0, 0] == 6.0
    assert right[1, 1] == 6.0
    assert normalized_total_variation(left, right) == pytest.approx(1.0)

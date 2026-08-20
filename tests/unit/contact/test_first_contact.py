"""Geometry-only Sphere First-Contact Normalization tests."""

from __future__ import annotations

import numpy as np
import pytest

from contact import (
    CandidateContactError,
    FirstContactSettings,
    canonical_sphere_alignment,
    find_first_contact,
    intersects,
    make_outer_compliant_surface,
)
from mesh.rigid.object import make_sphere_mesh
from model.fingertip_model import FingertipModel
from model.fingertip_model import FingertipParameters
from model.solid import build_fingertip_solid


@pytest.fixture(scope="module")
def contact_case():
    model = FingertipModel(FingertipParameters())
    solid = build_fingertip_solid(model)
    surface = make_outer_compliant_surface(solid)
    sphere = make_sphere_mesh(2.0, subdivisions=1)
    return model, surface, sphere


@pytest.fixture(scope="module")
def search_settings() -> FirstContactSettings:
    return FirstContactSettings(
        coarse_step_mm=0.25,
        tolerance_mm=1.0e-3,
        spawn_clearance_mm=0.05,
        max_travel_mm=20.0,
    )


def test_canonical_alignment_is_geometry_defined_and_collision_free(contact_case) -> None:
    model, surface, sphere = contact_case
    alignment = canonical_sphere_alignment(model, sphere, initial_gap_mm=0.25)

    assert alignment.target_point_mm[2] == 0.0
    assert np.isclose(np.linalg.norm(alignment.outward_normal), 1.0)
    np.testing.assert_allclose(
        np.asarray(alignment.approach_direction),
        -np.asarray(alignment.outward_normal),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert alignment.radius_mm == 2.0
    assert not intersects(surface, sphere, alignment.nominal_pose)


def test_first_contact_preserves_clear_hit_and_spawn_contract(contact_case, search_settings) -> None:
    model, surface, sphere = contact_case
    alignment = canonical_sphere_alignment(model, sphere)
    result = find_first_contact(
        surface,
        sphere,
        alignment.nominal_pose,
        alignment.approach_direction,
        search_settings,
    )

    assert not intersects(surface, sphere, result.clear_pose)
    assert intersects(surface, sphere, result.hit_pose)
    assert result.bracket_width_mm <= search_settings.tolerance_mm
    assert not intersects(surface, sphere, result.spawn_pose)
    assert abs(result.travel_to_contact_mm - alignment.initial_gap_mm) <= (
        search_settings.tolerance_mm
    )


def test_first_contact_is_deterministic(contact_case, search_settings) -> None:
    model, surface, sphere = contact_case
    alignment = canonical_sphere_alignment(model, sphere)
    first = find_first_contact(
        surface,
        sphere,
        alignment.nominal_pose,
        alignment.approach_direction,
        search_settings,
    )
    second = find_first_contact(
        surface,
        sphere,
        alignment.nominal_pose,
        alignment.approach_direction,
        search_settings,
    )

    assert first == second


def test_first_contact_is_invariant_to_free_space_start_distance(contact_case, search_settings) -> None:
    model, surface, sphere = contact_case
    near = canonical_sphere_alignment(model, sphere, initial_gap_mm=1.0)
    far = canonical_sphere_alignment(model, sphere, initial_gap_mm=10.0)
    near_result = find_first_contact(
        surface,
        sphere,
        near.nominal_pose,
        near.approach_direction,
        search_settings,
    )
    far_result = find_first_contact(
        surface,
        sphere,
        far.nominal_pose,
        far.approach_direction,
        search_settings,
    )

    np.testing.assert_allclose(
        near_result.contact_pose.translation_mm,
        far_result.contact_pose.translation_mm,
        atol=search_settings.tolerance_mm,
        rtol=0.0,
    )
    assert far_result.travel_to_contact_mm - near_result.travel_to_contact_mm == pytest.approx(
        9.0,
        abs=2.0 * search_settings.tolerance_mm,
    )


def test_first_contact_rejects_overlapping_reference_and_unreachable_search(contact_case) -> None:
    model, surface, sphere = contact_case
    alignment = canonical_sphere_alignment(model, sphere)
    direction = np.asarray(alignment.approach_direction)
    overlap_translation = np.asarray(alignment.nominal_pose.translation_mm) + 2.1 * direction
    overlap_pose = type(alignment.nominal_pose)(
        translation_mm=tuple(overlap_translation),
        quaternion_xyzw=alignment.nominal_pose.quaternion_xyzw,
    )
    with pytest.raises(ValueError, match="already intersects"):
        find_first_contact(
            surface,
            sphere,
            overlap_pose,
            alignment.approach_direction,
            FirstContactSettings(0.25, 1.0e-3, 0.05, 20.0),
        )

    with pytest.raises(CandidateContactError, match="exceeded max_travel_mm"):
        find_first_contact(
            surface,
            sphere,
            alignment.nominal_pose,
            alignment.approach_direction,
            FirstContactSettings(0.25, 1.0e-3, 0.05, 0.1),
        )

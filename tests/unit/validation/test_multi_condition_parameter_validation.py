"""Focused contract tests for contact-condition validation."""

from __future__ import annotations

import numpy as np

from contact import sphere_alignment_at_normalized_location
from mesh.rigid_object import make_sphere_mesh
from model import Fingertip
from validation.contact.multi_condition_parameter_validation import (
    COMPATIBLE_RADII_MM,
    ContactCondition,
    NORMALIZED_LOCATIONS,
    POST_CONTACT_DEPTHS_MM,
    _contact_geometry,
    _morphology_fingerprint,
    cell_end_clearance_mm,
    condition_identity,
    frozen_mechanics_contract,
    validation_morphologies,
)


def _geometry(morphology_id: str, location: float, radius: float, depth: float = 0.75):
    parameters = validation_morphologies()[morphology_id]
    fingerprint = _morphology_fingerprint(Fingertip(parameters))
    return _contact_geometry(
        morphology_id=morphology_id,
        parameters=parameters,
        condition=ContactCondition(location, radius, depth),
        morphology_fingerprint=fingerprint,
    )


def test_radius_reaches_mesh_and_alignment() -> None:
    row = _geometry("production_nominal", 0.5, 4.0)
    assert row["sphere_mesh_radius_mm"] == 4.0
    assert row["sphere_radius_mm"] == 4.0


def test_depth_propagation_uses_frozen_load_steps() -> None:
    assert frozen_mechanics_contract(0.75)["load_steps"] == 15
    assert frozen_mechanics_contract(1.50)["load_steps"] == 30


def test_depth_does_not_change_first_contact() -> None:
    shallow = _geometry("production_nominal", 0.5, 4.0, 0.75)
    deep = _geometry("production_nominal", 0.5, 4.0, 1.50)
    np.testing.assert_allclose(shallow["target_point_mm"], deep["target_point_mm"])
    np.testing.assert_allclose(shallow["outward_normal"], deep["outward_normal"])
    np.testing.assert_allclose(
        shallow["first_contact_pose_mm"], deep["first_contact_pose_mm"]
    )
    assert shallow["first_contact_travel_mm"] == deep["first_contact_travel_mm"]


def test_radius_preserves_surface_target_and_normal() -> None:
    small = _geometry("production_nominal", 0.25, 4.0)
    large = _geometry("production_nominal", 0.25, 5.0)
    np.testing.assert_allclose(small["target_point_mm"], large["target_point_mm"])
    np.testing.assert_allclose(small["outward_normal"], large["outward_normal"])


def test_radius_changes_nominal_sphere_pose_by_one_mm() -> None:
    small = _geometry("production_nominal", 0.25, 4.0)
    large = _geometry("production_nominal", 0.25, 5.0)
    delta = np.asarray(large["nominal_pose_mm"]) - np.asarray(small["nominal_pose_mm"])
    normal = np.asarray(small["outward_normal"])
    np.testing.assert_allclose(delta, normal, atol=1.0e-10, rtol=0.0)


def test_location_changes_physical_target() -> None:
    targets = {
        tuple(_geometry("production_nominal", location, 4.0)["target_point_mm"])
        for location in NORMALIZED_LOCATIONS
    }
    assert len(targets) == 3


def test_same_location_is_recomputed_for_second_morphology() -> None:
    first = _geometry("production_nominal", 0.5, 4.0)
    second = _geometry("shallow_wide_probe", 0.5, 4.0)
    assert not np.allclose(first["target_point_mm"], second["target_point_mm"])
    assert first["morphology_fingerprint"] != second["morphology_fingerprint"]


def test_finite_cell_rejects_six_mm_before_mechanics() -> None:
    assert cell_end_clearance_mm(4.0) > 0.0
    assert cell_end_clearance_mm(5.0) > 0.0
    row = _geometry("production_nominal", 0.5, 6.0)
    assert row["geometry_valid"] is True
    assert row["contact_valid"] is False
    assert row["failure_class"] == "CURRENT_DOMAIN_INCOMPATIBLE"
    assert row["mechanics_status"] == "not_run"


def test_condition_identity_contains_radius_and_depth() -> None:
    fingerprint = "morphology"
    location = ContactCondition(0.5, 4.0, 0.75)
    other_radius = ContactCondition(0.5, 5.0, 0.75)
    other_depth = ContactCondition(0.5, 4.0, 1.50)
    assert len({
        condition_identity(fingerprint, location),
        condition_identity(fingerprint, other_radius),
        condition_identity(fingerprint, other_depth),
    }) == 3


def test_frozen_mechanics_values_are_not_retuned() -> None:
    contract = frozen_mechanics_contract(1.50)
    assert contract["max_load_increment_mm"] == 0.05
    assert contract["vbd_iterations"] == 10
    assert contract["dt_s"] == 1.0e-3
    assert tuple(COMPATIBLE_RADII_MM) == (4.0, 5.0)


def test_alignment_center_uses_radius_plus_gap() -> None:
    tip = Fingertip(validation_morphologies()["production_nominal"])
    sphere = make_sphere_mesh(4.0, subdivisions=3)
    alignment = sphere_alignment_at_normalized_location(tip.geometry, sphere, 0.5)
    center = np.asarray(alignment.nominal_pose.translation_mm)
    target = np.asarray(alignment.target_point_mm)
    normal = np.asarray(alignment.outward_normal)
    np.testing.assert_allclose(center - target, 4.25 * normal, atol=1.0e-10, rtol=0.0)


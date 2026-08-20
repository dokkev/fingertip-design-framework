"""Normalized multi-location sphere-alignment contracts."""

from __future__ import annotations

import numpy as np
import pytest

from contact import (
    canonical_sphere_alignment,
    sphere_alignment_at_normalized_location,
)
from mesh.rigid.object import make_sphere_mesh
from finger.fingertip_geometry import FingertipModel
from finger.fingertip_parameters import FingertipParameters


@pytest.fixture(scope="module")
def model() -> FingertipModel:
    return FingertipModel(FingertipParameters())


@pytest.fixture(scope="module")
def sphere():
    return make_sphere_mesh(5.0, subdivisions=3)


def test_locations_use_native_arc_and_local_normal(model, sphere) -> None:
    alignments = tuple(
        sphere_alignment_at_normalized_location(
            model,
            location,
            radius_mm=5.0,
        )
        for location in (0.25, 0.50, 0.75)
    )

    assert [alignment.normalized_location for alignment in alignments] == [
        0.25,
        0.5,
        0.75,
    ]
    for alignment in alignments:
        outward = np.asarray(alignment.outward_normal)
        approach = np.asarray(alignment.approach_direction)
        np.testing.assert_allclose(approach, -outward, atol=1.0e-12, rtol=0.0)
        np.testing.assert_allclose(np.linalg.norm(outward), 1.0, atol=1.0e-12, rtol=0.0)
        center = np.asarray(alignment.nominal_pose.translation_mm)
        target = np.asarray(alignment.target_point_mm)
        np.testing.assert_allclose(
            center,
            target + (alignment.radius_mm + alignment.initial_gap_mm) * outward,
            atol=1.0e-10,
            rtol=0.0,
        )


def test_canonical_alignment_is_the_arc_midpoint(model, sphere) -> None:
    canonical = canonical_sphere_alignment(model, radius_mm=5.0)
    midpoint = sphere_alignment_at_normalized_location(
        model,
        0.5,
        radius_mm=5.0,
    )
    assert canonical == midpoint


def test_location_validation_is_fail_closed(model, sphere) -> None:
    with pytest.raises(ValueError, match="normalized_location"):
        sphere_alignment_at_normalized_location(model, -0.1, radius_mm=5.0)
    with pytest.raises(ValueError, match="normalized_location"):
        sphere_alignment_at_normalized_location(model, 1.1, radius_mm=5.0)

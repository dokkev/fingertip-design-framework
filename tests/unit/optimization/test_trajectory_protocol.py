from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from lumo.optimization.objectives import (
    TrajectoryObservation,
    compute_trajectory_objective,
    normalized_field_distance,
)
from lumo.optimization.protocol import DEFAULT_TRAJECTORY_PROTOCOL, TrajectoryEvaluationProtocol


def test_default_protocol_counts_and_absolute_depths() -> None:
    protocol = DEFAULT_TRAJECTORY_PROTOCOL
    assert protocol.trajectory_count == 6
    assert protocol.checkpoint_count == 3
    assert protocol.optical_state_count == 18
    assert protocol.checkpoint_depths_mm == pytest.approx((0.5, 1.0, 1.5))
    assert protocol.normalized_indentation_ratios(4.0) == pytest.approx((0.125, 0.25, 0.375))
    assert protocol.normalized_indentation_ratios(5.0) == pytest.approx((0.1, 0.2, 0.3))


def test_custom_protocol_changes_only_protocol_derived_values() -> None:
    protocol = TrajectoryEvaluationProtocol(
        contact_locations_u=(0.20, 0.50, 0.80),
        indenter_radii_mm=(3.5, 4.5, 5.0),
        checkpoint_depths_mm=(0.5, 1.0, 1.5, 2.0),
    )
    assert protocol.trajectory_count == 9
    assert protocol.checkpoint_count == 4
    assert protocol.optical_state_count == 36
    assert protocol.checkpoint_depths_mm == pytest.approx((0.5, 1.0, 1.5, 2.0))


def test_protocol_fingerprint_is_sensitive_to_every_field() -> None:
    base = DEFAULT_TRAJECTORY_PROTOCOL
    variants = (
        TrajectoryEvaluationProtocol((0.2, 0.5, 0.75), base.indenter_radii_mm, base.checkpoint_depths_mm),
        TrajectoryEvaluationProtocol(base.contact_locations_u, (4.0, 5.5), base.checkpoint_depths_mm),
        TrajectoryEvaluationProtocol(base.contact_locations_u, base.indenter_radii_mm, (0.5, 1.25, 1.5)),
        TrajectoryEvaluationProtocol(base.contact_locations_u, base.indenter_radii_mm, (0.5, 1.0)),
        TrajectoryEvaluationProtocol(base.contact_locations_u, base.indenter_radii_mm, base.checkpoint_depths_mm, initial_gap_mm=0.3),
    )
    assert all(variant.fingerprint != base.fingerprint for variant in variants)


def test_protocol_validation_does_not_require_default_counts() -> None:
    TrajectoryEvaluationProtocol((0.5,), (4.0,), (1.0,))
    with pytest.raises(ValueError):
        TrajectoryEvaluationProtocol((0.5,), (4.0,), (0.5, 0.5))


@pytest.mark.parametrize(
    ("location", "radius", "depth", "message"),
    (
        (-0.1, 4.0, 1.0, "location_u"),
        (1.1, 4.0, 1.0, "location_u"),
        (0.5, 0.0, 1.0, "radius_mm"),
        (0.5, 4.0, 0.0, "checkpoint_depth_mm"),
    ),
)
def test_trajectory_observation_rejects_nonphysical_labels(
    location,
    radius,
    depth,
    message,
) -> None:
        with pytest.raises(ValueError, match=message):
            TrajectoryObservation(location, radius, depth, np.ones(2), 1.0, 1.0)


def test_objective_compares_all_cross_location_states_and_radius_nuisance() -> None:
    observations = []
    for location in (0.25, 0.50):
        for radius in (4.0, 5.0):
            for depth in (0.5, 1.0):
                field = np.zeros((2, 2), dtype=float)
                field[int(location == 0.5), int(depth == 1.0)] = radius
                observations.append(
                    TrajectoryObservation(location, radius, depth, field, 1.0, 1.0)
                )
    result = compute_trajectory_objective(observations)
    assert result.d_inter is not None and result.d_inter >= 0.0
    assert result.d_radius is not None and result.d_radius >= 0.0
    assert result.objective_value == pytest.approx(result.d_inter - result.d_radius)


def test_one_radius_has_zero_radius_nuisance_term() -> None:
    observations = [
        TrajectoryObservation(0.25, 5.0, 1.0, np.array([1.0, 0.0]), 1.0, 1.0),
        TrajectoryObservation(0.50, 5.0, 1.0, np.array([0.0, 1.0]), 1.0, 1.0),
    ]
    result = compute_trajectory_objective(observations)
    assert result.d_radius == 0.0
    assert result.d_inter == normalized_field_distance(observations[0].field, observations[1].field)


def test_optimization_import_does_not_initialize_heavy_runtimes() -> None:
    code = """
import sys
import lumo.optimization
assert 'newton' not in sys.modules
assert 'warp' not in sys.modules
assert 'cupy' not in sys.modules
assert 'optix' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

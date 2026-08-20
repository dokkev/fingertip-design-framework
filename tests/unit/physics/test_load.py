"""Neutral particle-load contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from mechanics3d import ParticleLoad


def test_particle_load_is_read_only_and_reports_resultant() -> None:
    load = ParticleLoad(
        vertex_indices=np.asarray([3, 7], dtype=np.int32),
        forces_n=np.asarray([[1.0, 2.0, 0.0], [-1.0, 0.5, 3.0]]),
        load_steps=4,
    )

    np.testing.assert_allclose(load.resultant_force_n, [0.0, 2.5, 3.0])
    assert not load.vertex_indices.flags.writeable
    assert not load.forces_n.flags.writeable
    assert load.load_steps == 4


@pytest.mark.parametrize(
    "indices, forces",
    [
        ([1, 1], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ([1], [[np.nan, 0.0, 0.0]]),
        ([1], [[1.0, 0.0]]),
    ],
)
def test_particle_load_rejects_invalid_force_contract(indices, forces) -> None:
    with pytest.raises(ValueError):
        ParticleLoad(np.asarray(indices), np.asarray(forces))


def test_zero_particle_load_is_explicit() -> None:
    load = ParticleLoad.zero(load_steps=3)

    assert load.vertex_indices.shape == (0,)
    assert load.forces_n.shape == (0, 3)
    np.testing.assert_array_equal(load.resultant_force_n, [0.0, 0.0, 0.0])
    assert load.load_steps == 3

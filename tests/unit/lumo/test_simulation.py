from __future__ import annotations

import numpy as np

import lumo.simulation as simulation_module
from lumo.simulation import LumoSimulation


def test_checkpoint_values_are_absolute_depths_with_derived_annotations() -> None:
    fractions, ratios = LumoSimulation._checkpoint_values((0.5, 1.0, 1.5), 5.0)

    assert fractions == (1.0 / 3.0, 2.0 / 3.0, 1.0)
    assert ratios == (0.1, 0.2, 0.3)


def test_optix_runtime_is_created_once_and_reused(monkeypatch) -> None:
    simulation = object.__new__(LumoSimulation)
    simulation.optix_runtime = None
    created: list[object] = []
    runtime = object()

    def create_runtime():
        created.append(runtime)
        return runtime

    monkeypatch.setattr(simulation_module, "create_runtime", create_runtime)

    assert simulation._runtime() is runtime
    assert simulation._runtime() is runtime
    assert created == [runtime]


def test_checkpoint_values_reject_non_monotonic_or_non_finite_depths() -> None:
    with np.testing.assert_raises(ValueError):
        LumoSimulation._checkpoint_values((0.5, 0.5), 5.0)
    with np.testing.assert_raises(ValueError):
        LumoSimulation._checkpoint_values((0.5, float("nan")), 5.0)

"""Focused tests for the GUI's model-owned nominal initialization."""

from __future__ import annotations

from gui.design_space_app import _initial_state


def test_initial_state_uses_fingertip_nominal_defaults() -> None:
    state = _initial_state()
    geometry = state["geometry"]
    assert isinstance(geometry, dict)

    assert (
        geometry["flat_pad_width"],
        geometry["flat_pad_height"],
        geometry["semielliptical_pad_height"],
        geometry["stem_width"],
        geometry["stem_height"],
        geometry["void_width"],
        geometry["void_height"],
    ) == (30.0, 5.0, 9.0, 7.6, 6.0, 1.0, 0.0)

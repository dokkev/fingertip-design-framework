from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from fem import solve
from mesh import FingertipMesh, PadMesh, mesh_settings_for_level
from mesh.indenter import IndenterSettings
from model import Fingertip, FingertipParameters


def _mesh(tip: Fingertip) -> FingertipMesh:
    mesh = object.__new__(FingertipMesh)
    object.__setattr__(mesh, "parameters", tip.parameters)
    object.__setattr__(mesh, "settings", mesh_settings_for_level("medium"))
    object.__setattr__(
        mesh,
        "pad",
        PadMesh.from_arrays(
            node_ids=np.asarray([1, 2, 3], dtype=np.int64),
            reference_coordinates_mm=np.asarray(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=float,
            ),
            element_connectivity_node_ids=np.asarray([[1, 2, 3]], dtype=np.int64),
        ),
    )
    return mesh


@pytest.mark.parametrize("surface_x_mm", (0.0, 2.5))
def test_solve_routes_physical_surface_x_to_normal_fixture(monkeypatch, surface_x_mm) -> None:
    solve_module = importlib.import_module("fem.solve")
    tip = Fingertip(FingertipParameters())
    mesh = _mesh(tip)
    settings = IndenterSettings(radius_mm=3.25)
    captured = {}
    fixture = SimpleNamespace(settings=settings)

    def fake_fixture_builder(model, requested_x, requested_settings):
        captured["model"] = model
        captured["surface_x_mm"] = requested_x
        captured["settings"] = requested_settings
        return fixture

    def fake_runner(*args, **kwargs):
        captured["fixture_override"] = kwargs["fixture_override"]
        captured["indenter_settings"] = kwargs.get("indenter_settings")
        return {"solve_status": "FAIL", "final": {}}, None

    monkeypatch.setattr(
        solve_module,
        "build_normal_indenter_fixture_at_x",
        fake_fixture_builder,
    )
    monkeypatch.setattr(solve_module, "run_indentation_case", fake_runner)

    result = solve(
        tip,
        mesh,
        indentation=0.5,
        surface_x_mm=surface_x_mm,
        indenter=settings,
    )

    assert result.converged is False
    assert captured["model"] is tip.geometry
    assert captured["surface_x_mm"] == surface_x_mm
    assert captured["settings"] is settings
    assert captured["fixture_override"] is fixture
    assert captured["indenter_settings"] is None

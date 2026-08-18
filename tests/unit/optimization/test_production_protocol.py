"""Focused synthetic checks for the production morphology protocol."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from mesh import mesh_settings_for_level
from model import FingertipParameters, LED, OpticalMaterial
from optics import IndenterOptics
from optics.transport3d import Transport3DSettings
from optimization.design_space import DesignSpace, DesignVariable
from optimization.evaluator import DesignEvaluator
from optimization.study import OptimizationStudy
from optimization.scenarios import ScenarioGrid


class _FakeTransport:
    launched_weight = 1.0
    escaped_weight = 1.0
    absorbed_weight = 0.0
    terminated_weight = 0.0
    outgoing_surface_weight = 1.0
    object_absorbed_weight = 0.0
    object_transmitted_weight = 0.0
    object_interface_incident_weight = 0.0
    object_reflected_weight = 0.0
    energy_balance_error = 0.0
    geometry_metadata = {}
    internal_z_integrated_path_density = None
    internal_path_x_edges_mm = None
    internal_path_y_edges_mm = None

    def __init__(self, depth: float | None = None) -> None:
        self.depth = depth

    def lateral_outgoing_profiles(self):
        edges = np.array([0.0, 1.0])
        if self.depth is None:
            return edges, np.array([1.0]), np.array([1.0])
        return edges, np.array([1.0 + self.depth]), np.array([1.0])


class _FakeState:
    def __init__(self, depth: float) -> None:
        self.depth_mm = depth
        self.deformed_mesh = self
        self.indenter_pose = object()
        self.reaction_force_n = depth
        self.contact = {"external_pad_indenter": {"active_slave_node_ids": [1]}}
        self.active_external_node_ids = (1,)
        self.active_internal_node_ids = {"internal_left": (2,)}
        self.details = {
            "external_contact_width": {
                "active_node_count": int(depth * 10.0),
                "active_edge_count": 2,
                "chord_width_mm": depth + 0.1,
                "arc_length_mm": depth + 0.2,
            }
        }


class _FakeFEA:
    converged = True

    def __init__(self) -> None:
        self.states = {depth: _FakeState(depth) for depth in (0.5, 1.0, 1.5, 2.0)}

    def captured_state(self, depth: float) -> _FakeState:
        return self.states[depth]


def test_design_space_has_four_active_variables_and_derived_height() -> None:
    nominal = FingertipParameters()
    variables = tuple(
        DesignVariable(name, True, getattr(nominal, name), getattr(nominal, name))
        for name in ("flat_pad_height", "stem_width", "stem_height", "void_width")
    )
    space = DesignSpace(nominal, variables)
    decoded = space.decode(
        {
            "flat_pad_height": nominal.flat_pad_height,
            "stem_width": nominal.stem_width,
            "stem_height": nominal.stem_height,
            "void_width": nominal.void_width,
        }
    )
    assert tuple(variable.name for variable in space.variables) == (
        "flat_pad_height",
        "stem_width",
        "stem_height",
        "void_width",
    )
    assert decoded.flat_pad_width == 30.0
    assert decoded.semielliptical_pad_height == 9.0
    assert decoded.void_height == 0.0


def _production_study(*, indenter_optics: IndenterOptics | None) -> OptimizationStudy:
    nominal = FingertipParameters()
    variables = tuple(
        DesignVariable(
            name,
            True,
            getattr(nominal, name) - 0.1,
            getattr(nominal, name) + 0.1,
        )
        for name in ("flat_pad_height", "stem_width", "stem_height", "void_width")
    )
    return OptimizationStudy(
        design_space=DesignSpace(nominal, variables),
        scenario_grid=ScenarioGrid(),
        mesh_settings=mesh_settings_for_level("medium"),
        trace_settings=Transport3DSettings(mode="planar"),
        led=LED(),
        optical=OpticalMaterial(),
        indenter_optics=indenter_optics,
    )


def test_production_evaluator_rejects_missing_indenter_optics() -> None:
    with pytest.raises(ValueError, match="explicit indenter_optics"):
        DesignEvaluator(
            ScenarioGrid(),
            mesh_settings=mesh_settings_for_level("medium"),
            trace_settings=Transport3DSettings(mode="planar"),
            indenter_optics=None,  # type: ignore[arg-type]
        )


def test_production_study_rejects_missing_indenter_optics() -> None:
    with pytest.raises(ValueError, match="explicit indenter_optics"):
        _production_study(indenter_optics=None)


def test_evaluator_uses_one_reference_twelve_fea_and_48_loaded_traces(monkeypatch) -> None:
    import optimization.evaluator as evaluator_module

    trace_calls: list[tuple[object, dict[str, object]]] = []
    solve_calls: list[object] = []
    expected_poses: list[object] = []
    expected_contact_widths: list[dict[str, object]] = []

    def fake_trace(tip, mesh, **kwargs):
        trace_calls.append((mesh, kwargs))
        return _FakeTransport(
            None if len(trace_calls) == 1 else mesh.depth_mm
        )

    def fake_solve(*args, **kwargs):
        solve_calls.append(kwargs)
        fea = _FakeFEA()
        expected_poses.extend(state.indenter_pose for state in fea.states.values())
        expected_contact_widths.extend(
            state.details["external_contact_width"] for state in fea.states.values()
        )
        return fea

    monkeypatch.setattr(evaluator_module.Fingertip, "mesh", lambda self, settings: object())
    monkeypatch.setattr(evaluator_module, "trace_3d", fake_trace)
    monkeypatch.setattr(evaluator_module, "solve", fake_solve)

    parameters = FingertipParameters()
    before = asdict(parameters)
    indenter_optics = IndenterOptics("absorber")
    evaluator = DesignEvaluator(
        ScenarioGrid(),
        mesh_settings=mesh_settings_for_level("medium"),
        trace_settings=Transport3DSettings(mode="planar"),
        indenter_optics=indenter_optics,
    )
    result = evaluator.evaluate(parameters)

    assert result.status == "success"
    assert len(solve_calls) == 12
    assert len(trace_calls) == 49
    assert len(result.trajectories) == 12
    assert len(result.states) == 48
    assert result.minimum_auc == 1.0
    assert result.diagnostics["captured_state_count"] == 48
    assert asdict(parameters) == before
    assert "indenter_pose" not in trace_calls[0][1]
    assert "indenter_optics" not in trace_calls[0][1]
    assert [call[1]["indenter_pose"] for call in trace_calls[1:]] == expected_poses
    assert all(
        call[1]["indenter_optics"] is indenter_optics
        for call in trace_calls[1:]
    )
    assert [
        state.contact_diagnostics["external_contact_width"]
        for state in result.states
    ] == expected_contact_widths
    assert result.states[0].contact_diagnostics["active_external_node_ids"] == (1,)
    assert result.states[0].contact_diagnostics["contact_groups"] == {
        "external_pad_indenter": {"active_slave_node_ids": (1,)}
    }
    assert (
        result.states[0].contact_diagnostics["exact_indenter_pose"]
        is expected_poses[0]
    )

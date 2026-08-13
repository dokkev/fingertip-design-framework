from __future__ import annotations

from types import SimpleNamespace

import pytest

import optimization.evaluator as evaluator_module
from fem import FEAResult
from mesh import mesh_settings_for_level
from optics import TraceSettings
from optics.cross_section.domain import CrossSectionOpticsError
from model import FingertipParameters
from optimization import DesignEvaluator, ScenarioGrid


def _evaluator(grid: ScenarioGrid) -> DesignEvaluator:
    return DesignEvaluator(
        grid,
        mesh_settings=mesh_settings_for_level("medium"),
        trace_settings=TraceSettings(
            ray_count=3,
            grid_width=16,
            grid_height=16,
            maximum_segment_count=32,
        ),
        fem_steps=7,
    )


def test_evaluator_reuses_reference_and_forwards_every_scenario(monkeypatch) -> None:
    grid = ScenarioGrid(
        locations_x_mm=(-1.0, 1.0),
        indentations_mm=(0.5, 1.0),
        indenter_radii_mm=(2.0, 4.0),
    )
    calls = {"mesh": 0, "trace": [], "solve": []}
    fake_tip = SimpleNamespace()

    def fake_fingertip(parameters):
        del parameters
        return fake_tip

    def fake_mesh(settings):
        calls["mesh"] += 1
        assert settings.level == "medium"
        return "reference_mesh"

    fake_tip.mesh = fake_mesh

    def fake_trace(tip, mesh, settings):
        del tip, settings
        calls["trace"].append(mesh)
        if mesh == "reference_mesh":
            return SimpleNamespace(state="reference")
        return SimpleNamespace(state=mesh.scenario)

    def fake_solve(tip, mesh, **kwargs):
        del tip, mesh
        scenario = (kwargs["surface_x_mm"], kwargs["indentation"], kwargs["indenter"].radius_mm)
        calls["solve"].append(scenario)
        return SimpleNamespace(
            converged=True,
            reaction_force=1.25,
            deformed_mesh=SimpleNamespace(scenario=scenario),
        )

    def fake_metrics(reference, loaded):
        del reference
        return {
            "field_difference": 0.2,
            "centroid_shift_mm": 0.1,
            "escaped_fraction_change": 0.0,
            "absorbed_fraction_change": 0.0,
        }

    monkeypatch.setattr(evaluator_module, "Fingertip", fake_fingertip)
    monkeypatch.setattr(evaluator_module, "trace", fake_trace)
    monkeypatch.setattr(evaluator_module, "solve", fake_solve)
    monkeypatch.setattr(evaluator_module, "evaluate_transport", fake_metrics)
    monkeypatch.setattr(
        evaluator_module,
        "field_difference",
        lambda first, second: 0.25,
    )

    result = _evaluator(grid).evaluate(FingertipParameters())

    assert result.status == "success"
    assert result.score == pytest.approx(0.25)
    assert calls["mesh"] == 1
    assert len(calls["solve"]) == 8
    assert len(calls["trace"]) == 9
    assert calls["trace"].count("reference_mesh") == 1
    assert set(calls["solve"]) == {
        (scenario.location_x_mm, scenario.indentation_mm, scenario.indenter_radius_mm)
        for scenario in grid.scenarios
    }


def test_evaluator_uses_minimum_pair_separability_without_detectability_weight(
    monkeypatch,
) -> None:
    grid = ScenarioGrid(
        locations_x_mm=(0.0, 1.0, 2.0),
        indentations_mm=(0.5,),
        indenter_radii_mm=(2.0,),
    )
    states = {0.0: 0.0, 1.0: 0.4, 2.0: 0.7}

    monkeypatch.setattr(
        evaluator_module,
        "Fingertip",
        lambda parameters: SimpleNamespace(mesh=lambda settings: "reference_mesh"),
    )
    monkeypatch.setattr(
        evaluator_module,
        "trace",
        lambda tip, mesh, settings: SimpleNamespace(
            state=-1.0
            if mesh == "reference_mesh"
            else states[mesh.scenario[0]]
        ),
    )
    monkeypatch.setattr(
        evaluator_module,
        "solve",
        lambda tip, mesh, **kwargs: SimpleNamespace(
            converged=True,
            reaction_force=1.0,
            deformed_mesh=SimpleNamespace(
                scenario=(kwargs["surface_x_mm"], kwargs["indentation"], 2.0)
            ),
        ),
    )
    monkeypatch.setattr(
        evaluator_module,
        "evaluate_transport",
        lambda reference, loaded: {
            "field_difference": 0.9 - loaded.state,
            "centroid_shift_mm": 0.0,
            "escaped_fraction_change": 0.0,
            "absorbed_fraction_change": 0.0,
        },
    )
    monkeypatch.setattr(
        evaluator_module,
        "field_difference",
        lambda first, second: abs(first.state - second.state),
    )

    result = _evaluator(grid).evaluate(FingertipParameters())

    assert result.status == "success"
    assert result.score == pytest.approx(0.3)
    assert result.minimum_separability == pytest.approx(0.3)
    assert result.mean_separability == pytest.approx(0.35)
    assert result.median_separability == pytest.approx(0.35)
    assert result.minimum_detectability == pytest.approx(0.2)
    assert result.limiting_pair == grid.adjacent_pairs[1]


def test_nonconverged_fea_is_a_failure_without_a_penalty(monkeypatch) -> None:
    grid = ScenarioGrid((0.0, 1.0), (0.5,), (2.0,))
    calls = {"solve": 0}
    monkeypatch.setattr(
        evaluator_module,
        "Fingertip",
        lambda parameters: SimpleNamespace(mesh=lambda settings: "mesh"),
    )
    monkeypatch.setattr(
        evaluator_module,
        "trace",
        lambda tip, mesh, settings: SimpleNamespace(state=0.0),
    )

    def fake_solve(tip, mesh, **kwargs):
        calls["solve"] += 1
        return SimpleNamespace(converged=False)

    monkeypatch.setattr(evaluator_module, "solve", fake_solve)

    result = _evaluator(grid).evaluate(FingertipParameters())

    assert result.status == "fea_failure"
    assert result.score is None
    assert calls["solve"] == 1
    assert result.failure_message is not None


def test_expected_optics_failure_is_classified(monkeypatch) -> None:
    grid = ScenarioGrid((0.0, 1.0), (0.5,), (2.0,))
    monkeypatch.setattr(
        evaluator_module,
        "Fingertip",
        lambda parameters: SimpleNamespace(mesh=lambda settings: "mesh"),
    )

    def fake_trace(tip, mesh, settings):
        if mesh == "loaded":
            raise CrossSectionOpticsError("synthetic optical failure")
        return SimpleNamespace(state=0.0)

    monkeypatch.setattr(evaluator_module, "trace", fake_trace)
    monkeypatch.setattr(
        evaluator_module,
        "solve",
        lambda tip, mesh, **kwargs: SimpleNamespace(
            converged=True,
            reaction_force=None,
            deformed_mesh="loaded",
        ),
    )

    result = _evaluator(grid).evaluate(FingertipParameters())

    assert result.status == "optics_failure"
    assert result.score is None


def test_unexpected_runtime_error_is_not_swallowed(monkeypatch) -> None:
    grid = ScenarioGrid((0.0, 1.0), (0.5,), (2.0,))
    monkeypatch.setattr(
        evaluator_module,
        "Fingertip",
        lambda parameters: SimpleNamespace(mesh=lambda settings: "mesh"),
    )
    monkeypatch.setattr(
        evaluator_module,
        "trace",
        lambda tip, mesh, settings: (_ for _ in ()).throw(RuntimeError("bug")),
    )

    with pytest.raises(RuntimeError, match="bug"):
        _evaluator(grid).evaluate(FingertipParameters())

"""Focused pure-contract checks for the zero-height bottom contact policy."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from fem.kratos_settings import indentation_contact_groups
from fem.solve import solve
from mesh.types import BoundaryEdge, MeshNode, MeshedContactPair, MeshSettings
from model import FingertipParameters
from model.fingertip_model import FingertipModel
from optimization.evaluator import DesignEvaluator
from validation.fem.initial_contact_contract import (
    analytic_bottom_gap,
    classify_early_bottom_contact,
    initial_contact_zero_load_status,
    mesh_bottom_gap_statistics,
)


def test_analytic_nominal_bottom_gap_is_zero() -> None:
    assert analytic_bottom_gap(FingertipModel(FingertipParameters())) == pytest.approx(
        0.0
    )


def test_standard_mesh_bottom_gap_statistics_are_zero() -> None:
    nodes = {
        1: MeshNode(1, -1.0, -6.0, "pad"),
        2: MeshNode(2, 1.0, -6.0, "pad"),
        3: MeshNode(3, -1.0, -6.0, "rigid_carrier"),
        4: MeshNode(4, 1.0, -6.0, "rigid_carrier"),
    }
    settings = MeshSettings(
        level="medium",
        bulk_target_size_mm=1.0,
        contact_boundary_target_size_mm=0.5,
    )
    mesh = SimpleNamespace(
        nodes=nodes,
        boundary_edges={
            "pad_cutout_bottom": (BoundaryEdge((1, 2), "pad"),),
            "stem_bottom": (BoundaryEdge((3, 4), "rigid_carrier"),),
        },
        contact_pairs=(
            MeshedContactPair(
                "bottom_contact",
                "pad_cutout_bottom",
                "stem_bottom",
                0.0,
                0.0,
            ),
        ),
        settings=settings,
    )
    report = mesh_bottom_gap_statistics(mesh)
    assert report["min_gap_mm"] == pytest.approx(0.0)
    assert report["max_gap_mm"] == pytest.approx(0.0)
    assert report["mean_gap_mm"] == pytest.approx(0.0)


def test_three_pairs_registers_the_bottom_pair() -> None:
    groups = indentation_contact_groups("three_pairs")
    assert ("internal_bottom", "PadCutoutBottom", "StemBottom") in groups


def test_zero_gap_zero_load_does_not_require_active_contact() -> None:
    report = initial_contact_zero_load_status(
        initial_gap_mm=0.0,
        active_condition_count=0,
        maximum_abs_lm_pressure=0.0,
        tolerance_mm=1.0e-7,
    )
    assert report["status"] == "PASS"
    assert report["active_not_required_at_zero_load"] is True


def test_early_contact_classifier_distinguishes_delayed_engagement() -> None:
    def point(step: int, active: int, gap: float, lm: float) -> dict:
        return {
            "step": step,
            "prescribed_indenter_travel_mm": step / 12.0,
            "active_set_converged": True,
            "contact_groups": {
                "internal_bottom": {
                    "active_condition_count": active,
                    "active_condition_ids": [step] if active else [],
                    "weighted_gap": {},
                    "lagrange_multiplier_contact_pressure": {"min": lm, "max": lm},
                    "signed_geometric_gap": {
                        "available": True,
                        "max_signed_gap_mm": gap,
                        "maximum_penetration_mm": 0.0,
                    },
                }
            },
        }

    valid = classify_early_bottom_contact(
        [point(1, 0, 0.0, 0.0), point(2, 1, 0.0, -0.1)],
        gap_tolerance_mm=1.0e-7,
    )
    delayed = classify_early_bottom_contact(
        [point(1, 0, 0.2, 0.0), point(2, 1, 0.0, -0.1)],
        gap_tolerance_mm=1.0e-7,
    )
    assert valid["delayed_engagement"] is False
    assert valid["first_active_step"] == 2
    assert valid["first_nonzero_lm_pressure_step"] == 2
    assert delayed["delayed_engagement"] is True


def test_low_level_solver_and_evaluator_defaults_remain_48_steps() -> None:
    assert inspect.signature(solve).parameters["steps"].default == 48
    assert inspect.signature(DesignEvaluator).parameters["fem_steps"].default == 48

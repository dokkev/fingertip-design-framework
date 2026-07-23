"""Gmsh-backed smoke tests for Phase 4I result extraction."""

from __future__ import annotations

import numpy as np
import pytest

from mesh.fingertip import generate_fingertip_mesh
from mesh.indenter import build_indenter_fixture
from fem.results import (
    compressive_indenter_reaction,
    contact_width_metrics,
    extract_outer_arc_profile,
    interpolate_profile,
    ordered_boundary_node_ids,
    pad_strain_det_f_statistics,
    profile_error_metrics,
    unique_projected_reaction,
)
from validation.common.io import write_indentation_history
from visualization.indentation import save_history_plots
from mesh.types import mesh_settings_for_level
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


@pytest.fixture(scope="module")
def meshed_model():
    model = FingertipModel(FingertipParameters())
    mesh = generate_fingertip_mesh(model, mesh_settings_for_level("medium"))
    return model, mesh


def test_reaction_sign_and_unique_node_summation() -> None:
    reactions = {1: (0.0, 2.0), 2: (0.0, 3.0)}
    assert unique_projected_reaction(reactions, [1, 1, 2], (0.0, 1.0)) == 5.0
    assert compressive_indenter_reaction(
        reactions, [1, 1, 2], (0.0, 1.0)
    ) == 5.0


def test_contact_width_uses_active_nodes_and_source_edges(meshed_model) -> None:
    model, mesh = meshed_model
    ordered = ordered_boundary_node_ids(
        mesh,
        "pad_outer_arc",
        model.boundaries.segments["pad_outer_arc"].geometry,
    )
    center = len(ordered) // 2
    active = ordered[center - 1 : center + 2]
    fixture = build_indenter_fixture(model)
    result = contact_width_metrics(mesh, active, fixture.frame.tangent)
    assert result["active_node_count"] == 3
    assert result["active_edge_count"] == 2
    assert result["chord_width_mm"] > 0.0
    assert result["arc_length_mm"] > 0.0


def test_outer_arc_profile_is_ordered_and_complete(meshed_model) -> None:
    model, mesh = meshed_model
    fixture = build_indenter_fixture(model)
    displacements = {node_id: (0.0, 0.0) for node_id in mesh.nodes}
    profile = extract_outer_arc_profile(
        model, mesh, displacements, fixture.frame
    )
    assert len(profile) == len(mesh.boundary_edges["pad_outer_arc"]) + 1
    assert profile[0]["normalized_arc_coordinate"] == 0.0
    assert profile[-1]["normalized_arc_coordinate"] == 1.0
    assert all(
        second["normalized_arc_coordinate"] > first["normalized_arc_coordinate"]
        for first, second in zip(profile, profile[1:])
    )
    assert all(record["local_normal_displacement_mm"] == 0.0 for record in profile)


def test_profile_interpolation_and_synthetic_convergence_metric() -> None:
    profile = [
        {
            "normalized_arc_coordinate": coordinate,
            "local_normal_displacement_mm": 2.0 * coordinate,
            "local_tangential_displacement_mm": -coordinate,
        }
        for coordinate in (0.0, 0.5, 1.0)
    ]
    interpolated = interpolate_profile(profile, (0.0, 0.25, 0.5, 0.75, 1.0))
    assert interpolated["normal_displacement_mm"] == pytest.approx(
        (0.0, 0.5, 1.0, 1.5, 2.0)
    )
    metric = profile_error_metrics(
        [value * 1.01 for value in interpolated["normal_displacement_mm"]],
        interpolated["normal_displacement_mm"],
        1.0e-5,
    )
    assert metric["relative_l2_error"] == pytest.approx(0.01)


def test_pad_strain_statistics_exclude_both_rigid_domains(meshed_model) -> None:
    _, mesh = meshed_model
    displacements = {
        node_id: ((100.0, -100.0) if node.domain == "rigid_carrier" else (0.0, 0.0))
        for node_id, node in mesh.nodes.items()
    }
    result = pad_strain_det_f_statistics(mesh, displacements)
    assert result["rigid_domains_excluded"]
    assert result["pad_element_count"] == len(mesh.pad_elements)
    assert result["det_f"]["min"] == pytest.approx(1.0)
    assert result["det_f"]["max"] == pytest.approx(1.0)
    assert result["maximum_absolute_green_lagrange_component"]["value"] == pytest.approx(0.0)


def test_external_only_history_outputs_do_not_require_internal_groups(
    tmp_path,
) -> None:
    def point(step: int) -> dict[str, object]:
        return {
            "step": step,
            "pseudo_time": float(step),
            "prescribed_indenter_travel_mm": 0.01 * step,
            "achieved_indentation_mm": 0.01 * step,
            "indenter_normal_reaction_n": 0.1 * step,
            "support_signed_reaction_along_loading_n": -0.1 * step,
            "force_equilibrium_error": 0.0,
            "nonlinear_iterations": 2,
            "solver_converged": True,
            "active_set_converged": True,
            "contact_groups": {
                "external_pad_indenter": {
                    "active_condition_count": 1,
                    "weighted_gap": {"min": -1.0e-6, "mean": -1.0e-7},
                    "signed_geometric_gap": {
                        "maximum_penetration_mm": 1.0e-6
                    },
                }
            },
            "external_contact_width": {
                "chord_width_mm": 0.1,
                "arc_length_mm": 0.1,
            },
            "pad_strain_det_f": {
                "maximum_principal_green_lagrange_strain": {
                    "value": 0.01
                },
                "det_f": {"min": 0.99},
            },
            "maximum_pad_displacement_mm": 0.01 * step,
            "volumetric_strain": {"min": -0.01, "max": 0.01},
            "solve_wall_clock_seconds": 0.1,
        }

    history = [point(1), point(2)]
    write_indentation_history(tmp_path / "history.csv", history)
    save_history_plots({"history": history}, tmp_path / "plots")
    assert (tmp_path / "history.csv").is_file()
    assert (tmp_path / "plots" / "contact_groups.png").is_file()

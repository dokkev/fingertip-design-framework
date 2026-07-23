"""Phase 4I-E mirror, orientation, and endpoint assembly contracts."""

from __future__ import annotations

import math

import pytest

from mesh.fingertip import generate_fingertip_mesh
from mesh.types import mesh_settings_for_level
from validation.fingertip.internal_contact.right_side_core import (
    ORIENTATION_VARIANTS,
    audit_side_orientation,
    boundary_orientation_contract,
    common_audit_mesh,
    left_right_mirror_contract,
    mesh_for_orientation_variant,
    reverse_boundary_condition_ordering,
)
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


@pytest.fixture(scope="module")
def medium_source():
    return common_audit_mesh("medium")


def _boundary_count(mesh) -> int:
    return sum(len(edges) for edges in mesh.boundary_edges.values())


def _undirected_connectivity(mesh, tag: str) -> set[tuple[int, int]]:
    return {
        tuple(sorted(edge.node_ids))
        for edge in mesh.boundary_edges[tag]
    }


def test_left_right_mirror_node_and_condition_mapping(
    medium_source,
) -> None:
    model, mesh = medium_source
    contract = left_right_mirror_contract(mesh, model)
    assert contract["checks"]["pad_nodes_mirror"]
    assert contract["checks"]["stem_nodes_mirror"]
    assert contract["checks"]["pad_conditions_mirror"]
    assert contract["checks"]["stem_conditions_mirror"]
    assert contract["pad_node_mapping"]
    assert contract["pad_condition_mapping"]


def test_physical_normals_obey_left_right_reflection(
    medium_source,
) -> None:
    model, mesh = medium_source
    contract = left_right_mirror_contract(mesh, model)
    assert contract["checks"]["physical_normals_reflect"]
    assert contract["checks"]["source_ordering_is_physical"]
    assert all(
        record["normal_error"] <= 1.0e-12
        for record in contract["normal_reflection_checks"]
    )


@pytest.mark.parametrize(
    ("variant_name", "reversed_tags"),
    (
        ("R10", {"pad_cutout_right"}),
        ("R01", {"stem_right"}),
        ("R11", {"pad_cutout_right", "stem_right"}),
    ),
)
def test_right_orientation_reversal_changes_only_requested_line_ordering(
    medium_source,
    variant_name: str,
    reversed_tags: set[str],
) -> None:
    _, mesh = medium_source
    changed = mesh_for_orientation_variant(
        mesh, ORIENTATION_VARIANTS[variant_name]
    )
    assert changed.quality.node_count == mesh.quality.node_count
    assert _boundary_count(changed) == _boundary_count(mesh)
    assert [element.node_ids for element in changed.elements] == [
        element.node_ids for element in mesh.elements
    ]
    for tag, original_edges in mesh.boundary_edges.items():
        changed_edges = changed.boundary_edges[tag]
        assert _undirected_connectivity(changed, tag) == (
            _undirected_connectivity(mesh, tag)
        )
        assert len(changed_edges) == len(
            _undirected_connectivity(changed, tag)
        )
        if tag in reversed_tags:
            assert [edge.node_ids for edge in changed_edges] == [
                tuple(reversed(edge.node_ids)) for edge in original_edges
            ]
        else:
            assert [edge.node_ids for edge in changed_edges] == [
                edge.node_ids for edge in original_edges
            ]


@pytest.mark.parametrize("mesh_level", ("medium", "fine"))
def test_physical_orientation_rule_is_mesh_refinement_independent(
    mesh_level: str,
) -> None:
    model = FingertipModel(FingertipParameters())
    mesh = generate_fingertip_mesh(
        model, mesh_settings_for_level(mesh_level)
    )
    original = boundary_orientation_contract(mesh, model)
    reversed_slave = boundary_orientation_contract(
        mesh_for_orientation_variant(mesh, ORIENTATION_VARIANTS["R10"]),
        model,
    )
    assert original["all_ordering_normals_physical"]
    assert not reversed_slave["all_ordering_normals_physical"]
    assert not reversed_slave["surfaces"]["pad_cutout_right"][
        "all_ordering_normals_physical"
    ]


def test_invalid_orientation_tag_is_rejected(medium_source) -> None:
    _, mesh = medium_source
    with pytest.raises(ValueError, match="unknown boundary orientation"):
        reverse_boundary_condition_ordering(mesh, ("not_a_surface",))


@pytest.fixture(scope="module")
def r00_audit():
    result, dof_rows, contact_records = audit_side_orientation(
        "right", "medium", ORIENTATION_VARIANTS["R00"]
    )
    return result, dof_rows, contact_records


def test_upper_endpoint_equation_id_and_lm_contributors_are_resolved(
    r00_audit,
) -> None:
    result, dof_rows, _ = r00_audit
    endpoint = result["diagnostic"]["endpoint_assembly"]
    equation_id = endpoint["lm_equation_id"]
    matching = [
        row
        for row in dof_rows
        if row["equation_id"] == equation_id and row["assembled"]
    ]
    assert len(matching) == 1
    assert matching[0]["node_id"] == endpoint["slave_endpoint_node_id"]
    assert (
        matching[0]["variable"]
        == "LAGRANGE_MULTIPLIER_CONTACT_PRESSURE"
    )
    contributors = endpoint["local_condition_contributors"]
    assert contributors
    assert all(
        record["global_lm_equation_id"] == equation_id
        for record in contributors
    )
    assert all(
        math.isfinite(record["local_row_norm_all_columns"])
        for record in contributors
    )


def test_r00_runtime_roles_and_pair_purity_are_preserved(
    r00_audit,
) -> None:
    result, _, _ = r00_audit
    contract = result["diagnostic"]["runtime_contact_contract"]
    right = contract["groups"]["internal_right"]
    assert right["slave"] == "PadCutoutRight"
    assert right["master"] == "StemRight"
    assert right["checks"]["all_slave_nodes_flagged_slave"]
    assert right["checks"]["all_master_nodes_flagged_master"]
    assert result["diagnostic"]["pair_purity"][
        "all_generated_conditions_pair_pure"
    ]


def test_r00_detects_invalid_extra_endpoint_projection(
    r00_audit,
) -> None:
    result, _, _ = r00_audit
    projections = result["diagnostic"]["endpoint_assembly"][
        "pairing_projection"
    ]
    assert len(projections) == 2
    assert sum(
        record["endpoint_projection"]["success"]
        for record in projections
    ) == 1
    assert result["diagnostic"]["endpoint_assembly"]["near_zero"]
    assert result["status"] == "FAIL"

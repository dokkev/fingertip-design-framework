from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Point

from mesh.types import BoundaryEdge, MeshNode, T3Element
from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters
from model.fingertip_sensor_model import FingertipSensorModel
from model.led_parameters import LEDParameters
from optics.adapters import (
    build_pad_field_from_arrays,
    build_pad_field_from_mesh_and_displacements,
    build_preview_pad_mesh_template,
    load_pad_field_npz,
)
from optics.cross_section import (
    CrossSectionTraceSettings,
    build_mesh_state_optical_domain,
    build_no_load_optical_domain,
    trace_cross_section_transport,
    trace_pad_state,
)
from optics.geometry import (
    ExtrudedOpticalMeshTemplate,
    PadDeformationState2D,
    PadMeshTemplate2D,
)
from optics.geometry.pad_mesh_template import InvalidPadMeshTemplate


@pytest.fixture
def square_mesh_arrays() -> dict[str, np.ndarray]:
    return {
        "node_ids": np.asarray([30, 10, 40, 20], dtype=np.int64),
        "reference_coordinates_mm": np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=float,
        ),
        "element_connectivity_node_ids": np.asarray(
            [[30, 40, 10], [30, 20, 40]],
            dtype=np.int64,
        ),
    }


def test_sensor_metadata_has_one_physical_owner_and_does_not_change_csg() -> None:
    geometry = FingertipModel(FingertipParameters())
    original = (
        geometry.outer_pad_geometry.area,
        geometry.pad_material_geometry.area,
        geometry.link_geometry.area,
        geometry.material_geometry.area,
        tuple(geometry.boundaries.segments),
    )

    sensor = FingertipSensorModel.from_geometry(geometry)

    assert sensor.led_package_geometry.bounds == pytest.approx(
        (-2.0, geometry.parameters.stem_tip_y, 2.0,
         geometry.parameters.stem_tip_y + 2.0)
    )
    assert sensor.led_source_position_2d == (
        0.0,
        geometry.parameters.stem_tip_y,
    )
    assert original == (
        geometry.outer_pad_geometry.area,
        geometry.pad_material_geometry.area,
        geometry.link_geometry.area,
        geometry.material_geometry.area,
        tuple(geometry.boundaries.segments),
    )


def test_mesh_template_normalizes_winding_and_validates_states(
    square_mesh_arrays: dict[str, np.ndarray],
) -> None:
    template = PadMeshTemplate2D.from_arrays(**square_mesh_arrays)

    np.testing.assert_array_equal(
        template.triangles,
        np.asarray([[0, 1, 2], [0, 2, 3]]),
    )
    assert {tuple(edge) for edge in template.boundary_edges} == {
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    }
    zero = PadDeformationState2D.zero(template)
    np.testing.assert_array_equal(
        template.coordinates_for(zero),
        template.reference_coordinates_mm,
    )
    loaded = PadDeformationState2D(
        np.asarray(
            [[0.0, 0.0], [0.1, 0.0], [0.1, 0.2], [0.0, 0.2]]
        )
    )
    np.testing.assert_allclose(
        template.coordinates_for(loaded),
        template.reference_coordinates_mm + loaded.displacement_mm,
    )
    inverted = PadDeformationState2D(
        np.asarray([[0.0, 0.0], [-2.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    )
    with pytest.raises(InvalidPadMeshTemplate, match="inverted"):
        template.validate_state(inverted)


def test_extrusion_preserves_faces_and_depth_between_states(
    square_mesh_arrays: dict[str, np.ndarray],
) -> None:
    template = PadMeshTemplate2D.from_arrays(**square_mesh_arrays)
    zero = PadDeformationState2D.zero(template)
    loaded = PadDeformationState2D(
        np.asarray(
            [[0.0, 0.0], [0.1, 0.0], [0.1, 0.2], [0.0, 0.2]]
        )
    )
    extrusion = ExtrudedOpticalMeshTemplate.from_pad_mesh(
        template,
        depth_mm=10.0,
    )
    original_faces = extrusion.faces_3d.copy()
    zero_vertices = extrusion.vertices_for_state(template, zero)
    loaded_vertices = extrusion.vertices_for_state(template, loaded)

    assert zero_vertices.shape == (2 * len(template.node_ids), 3)
    assert loaded_vertices.shape == zero_vertices.shape
    np.testing.assert_array_equal(extrusion.faces_3d, original_faces)
    np.testing.assert_allclose(zero_vertices[:, 2], [-5.0] * 4 + [5.0] * 4)
    np.testing.assert_array_equal(loaded_vertices[:, 2], zero_vertices[:, 2])
    assert not np.array_equal(loaded_vertices[:, :2], zero_vertices[:, :2])
    assert np.min(extrusion.faces_3d) >= 0
    assert np.max(extrusion.faces_3d) < len(zero_vertices)


def test_array_and_npz_adapters_converge(
    tmp_path,
    square_mesh_arrays: dict[str, np.ndarray],
) -> None:
    displacement = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [0.1, 0.2], [0.0, 0.2]],
        dtype=float,
    )
    direct = build_pad_field_from_arrays(
        **square_mesh_arrays,
        displacement_mm=displacement,
        metadata={"source": "arrays"},
    )
    artifact = tmp_path / "deformation.npz"
    np.savez(artifact, **square_mesh_arrays, displacement_mm=displacement)
    persisted = load_pad_field_npz(artifact, metadata={"source": "npz"})

    for name in (
        "node_ids",
        "reference_coordinates_mm",
        "triangles",
        "boundary_edges",
    ):
        np.testing.assert_array_equal(
            getattr(direct.template, name),
            getattr(persisted.template, name),
        )
    np.testing.assert_array_equal(
        direct.state.displacement_mm,
        persisted.state.displacement_mm,
    )


def test_source_power_scales_raw_transport_linearly() -> None:
    geometry = FingertipModel(
        FingertipParameters(void_width=1.0, void_height=2.0)
    )
    baseline_sensor = FingertipSensorModel.from_geometry(
        geometry,
        led=LEDParameters(relative_radiant_power=1.0),
    )
    doubled_sensor = FingertipSensorModel.from_geometry(
        geometry,
        led=LEDParameters(relative_radiant_power=2.0),
    )
    domain = build_no_load_optical_domain(baseline_sensor)
    settings = CrossSectionTraceSettings(
        ray_count=31,
        grid_width=48,
        grid_height=48,
        maximum_segment_count=5000,
    )
    baseline = trace_cross_section_transport(
        domain,
        led=baseline_sensor.led,
        material=baseline_sensor.optical_material,
        settings=settings,
    )
    doubled = trace_cross_section_transport(
        domain,
        led=doubled_sensor.led,
        material=doubled_sensor.optical_material,
        settings=settings,
    )

    np.testing.assert_allclose(
        doubled.weighted_path_density,
        2.0 * baseline.weighted_path_density,
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        [
            doubled.launched_weight,
            doubled.escaped_weight,
            doubled.absorbed_weight,
            doubled.terminated_weight,
        ],
        2.0
        * np.asarray(
            [
                baseline.launched_weight,
                baseline.escaped_weight,
                baseline.absorbed_weight,
                baseline.terminated_weight,
            ]
        ),
        rtol=1.0e-12,
        atol=1.0e-14,
    )


def test_semantic_boundary_partition_is_canonical_complete_and_immutable(
    square_mesh_arrays: dict[str, np.ndarray],
) -> None:
    semantic_edges = {
        "zeta": np.asarray([[40, 10], [20, 40]], dtype=np.int64),
        "alpha": np.asarray([[10, 30], [30, 20]], dtype=np.int64),
    }
    template = PadMeshTemplate2D.from_arrays(
        **square_mesh_arrays,
        boundary_edge_node_ids_by_tag=semantic_edges,
    )

    assert template.semantic_boundary_tags == ("alpha", "zeta")
    np.testing.assert_array_equal(
        template.boundary_edges_for("alpha"),
        np.asarray([[0, 1], [3, 0]]),
    )
    np.testing.assert_array_equal(
        template.boundary_edges_for("zeta"),
        np.asarray([[1, 2], [2, 3]]),
    )
    owned_edges = np.vstack(
        [
            template.boundary_edges_for(tag)
            for tag in template.semantic_boundary_tags
        ]
    )
    assert {tuple(edge) for edge in owned_edges} == {
        tuple(edge) for edge in template.boundary_edges
    }
    assert len(owned_edges) == len(template.boundary_edges)
    with pytest.raises(ValueError):
        template.boundary_edges_for("alpha")[0, 0] = 2
    with pytest.raises(TypeError):
        template.boundary_edges_by_tag["new"] = np.asarray([[0, 1]])

    duplicate_ownership = {
        "first": np.asarray([[30, 10], [10, 40]], dtype=np.int64),
        "second": np.asarray(
            [[10, 30], [40, 20], [20, 30]],
            dtype=np.int64,
        ),
    }
    with pytest.raises(InvalidPadMeshTemplate, match="belongs to both"):
        PadMeshTemplate2D.from_arrays(
            **square_mesh_arrays,
            boundary_edge_node_ids_by_tag=duplicate_ownership,
        )

    incomplete_partition = {
        "partial": np.asarray(
            [[30, 10], [10, 40], [40, 20]],
            dtype=np.int64,
        )
    }
    with pytest.raises(InvalidPadMeshTemplate, match="missing edges"):
        PadMeshTemplate2D.from_arrays(
            **square_mesh_arrays,
            boundary_edge_node_ids_by_tag=incomplete_partition,
        )


def _synthetic_loaded_preview() -> tuple[
    FingertipSensorModel,
    PadMeshTemplate2D,
    PadDeformationState2D,
    float,
]:
    geometry = FingertipModel(FingertipParameters())
    sensor = FingertipSensorModel.from_geometry(geometry)
    template = build_preview_pad_mesh_template(sensor)
    displacement_magnitude_mm = 0.25
    displacement = np.zeros_like(template.reference_coordinates_mm)
    bottom_nodes = template.boundary_node_indices_for("pad_cutout_bottom")
    displacement[bottom_nodes, 1] = -displacement_magnitude_mm
    state = PadDeformationState2D(
        displacement_mm=displacement,
        metadata={"condition": "synthetic_loaded"},
    )
    template.validate_state(state)
    return sensor, template, state, displacement_magnitude_mm


def test_preview_boundary_classification_partitions_every_edge() -> None:
    geometry = FingertipModel(FingertipParameters())
    sensor = FingertipSensorModel.from_geometry(geometry)
    template = build_preview_pad_mesh_template(sensor)
    expected_tags = {
        "pad_bond_left",
        "pad_bond_right",
        "pad_outer_left",
        "pad_outer_right",
        "pad_outer_arc",
        "pad_cutout_left",
        "pad_cutout_right",
        "pad_cutout_bottom",
    }

    assert expected_tags.issubset(template.semantic_boundary_tags)
    semantic_edges = np.vstack(
        [
            template.boundary_edges_for(tag)
            for tag in template.semantic_boundary_tags
        ]
    )
    assert len(semantic_edges) == len(template.boundary_edges)
    assert {tuple(edge) for edge in semantic_edges} == {
        tuple(edge) for edge in template.boundary_edges
    }


def test_loaded_outer_envelope_contains_air_gap_under_fixed_source() -> None:
    sensor, template, loaded_state, displacement_magnitude_mm = (
        _synthetic_loaded_preview()
    )
    loaded_domain = build_mesh_state_optical_domain(
        sensor,
        template,
        loaded_state,
    )
    source_x, source_y = sensor.led_source_position_2d
    probe = Point(source_x, source_y - 0.5 * displacement_magnitude_mm)

    assert loaded_domain.outer_envelope.covers(probe)
    assert loaded_domain.accessible_region.covers(probe)
    assert not loaded_domain.silicone_region.covers(probe)
    assert not loaded_domain.rigid_region.covers(probe)
    tolerance = sensor.geometry.parameters.geometry_tolerance
    assert loaded_domain.outer_envelope.buffer(tolerance).covers(
        loaded_domain.silicone_region
    )


def test_loaded_transport_starts_in_air_and_reaches_silicone_deterministically() -> None:
    sensor, template, loaded_state, _ = _synthetic_loaded_preview()
    settings = CrossSectionTraceSettings(
        ray_count=31,
        grid_width=48,
        grid_height=48,
        maximum_segment_count=5000,
    )

    first_domain, first = trace_pad_state(
        sensor,
        template,
        loaded_state,
        settings,
    )
    second_domain, second = trace_pad_state(
        sensor,
        template,
        loaded_state,
        settings,
    )

    assert first_domain.silicone_region.equals(second_domain.silicone_region)
    assert float(np.max(first.weighted_path_density)) > 0.0
    assert any(segment.medium == "air" for segment in first.segments)
    assert any(segment.medium == "silicone" for segment in first.segments)
    assert first.segments[0].medium == "air"
    assert first.launched_weight > 0.0
    np.testing.assert_allclose(
        first.escaped_weight
        + first.absorbed_weight
        + first.terminated_weight,
        first.launched_weight,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(
        first.weighted_path_density,
        second.weighted_path_density,
    )
    assert first.segments == second.segments


def test_in_memory_fea_adapter_preserves_only_pad_boundary_tags() -> None:
    nodes = {
        1: MeshNode(1, 0.0, 0.0, "pad"),
        2: MeshNode(2, 1.0, 0.0, "pad"),
        3: MeshNode(3, 1.0, 1.0, "pad"),
        4: MeshNode(4, 0.0, 1.0, "pad"),
        5: MeshNode(5, 2.0, 0.0, "rigid_carrier"),
        6: MeshNode(6, 2.0, 1.0, "rigid_carrier"),
    }
    mesh = SimpleNamespace(
        nodes=nodes,
        pad_elements=(
            T3Element(1, (1, 2, 3), "pad"),
            T3Element(2, (1, 3, 4), "pad"),
        ),
        boundary_edges={
            "pad_bottom": (BoundaryEdge((2, 1), "pad"),),
            "pad_right": (BoundaryEdge((3, 2), "pad"),),
            "pad_top": (BoundaryEdge((4, 3), "pad"),),
            "pad_left": (BoundaryEdge((1, 4), "pad"),),
            "stem_left": (BoundaryEdge((5, 6), "rigid_carrier"),),
        },
    )
    displacements = {
        node_id: (0.0, 0.0) for node_id in (1, 2, 3, 4)
    }

    field = build_pad_field_from_mesh_and_displacements(
        mesh,
        displacements,
    )

    assert field.template.semantic_boundary_tags == (
        "pad_bottom",
        "pad_left",
        "pad_right",
        "pad_top",
    )
    assert "stem_left" not in field.template.semantic_boundary_tags
    owned_edges = np.vstack(
        [
            field.template.boundary_edges_for(tag)
            for tag in field.template.semantic_boundary_tags
        ]
    )
    assert {tuple(edge) for edge in owned_edges} == {
        tuple(edge) for edge in field.template.boundary_edges
    }

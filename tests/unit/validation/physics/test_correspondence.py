"""Synthetic descriptor tests independent of the local FEA corpus."""

from __future__ import annotations

import numpy as np
import pytest

from lumo.physics import PreparedFingertipMesh, NewtonResult, TetMeshData
from validation.physics.correspondence import compare_mechanics_states
from validation.reference.kratos3d.fea3d_reference import FEA3DReferenceState


def _synthetic_states():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    tetrahedra = np.asarray(
        [[0, 1, 3, 4], [1, 2, 3, 6], [1, 3, 4, 6], [1, 4, 5, 6], [3, 4, 6, 7]],
        dtype=np.int32,
    )
    prepared = PreparedFingertipMesh(
        tet_mesh=TetMeshData(vertices, tetrahedra),
        source_node_ids=np.arange(8, dtype=np.int64),
        support_vertex_indices=(0, 1, 2, 3),
        surface_triangles={
            "void_left": np.asarray([[0, 3, 4]], dtype=np.int32),
            "void_right": np.asarray([[1, 2, 5]], dtype=np.int32),
            "void_bottom": np.asarray([[0, 1, 4]], dtype=np.int32),
            "outer_compliant_arc": np.asarray([[3, 2, 7]], dtype=np.int32),
        },
        morphology_fingerprint="synthetic",
    )
    fea_displacement = np.zeros((8, 3), dtype=np.float32)
    fea_displacement[[1, 2, 5, 6], 0] = 0.1
    vbd_displacement = np.zeros((8, 3), dtype=np.float32)
    vbd_displacement[[1, 2, 5, 6], 0] = 0.12
    reference = FEA3DReferenceState(
        source_path="synthetic.json",
        reference_coordinates_mm=vertices.astype(np.float64),
        deformed_coordinates_mm=vertices.astype(np.float64) + fea_displacement.astype(np.float64),
        displacement_mm=fea_displacement.astype(np.float64),
        source_node_ids=np.arange(8, dtype=np.int64),
        morphology_fingerprint="synthetic",
        load_metadata={"load": {"center_x_mm": 0.5, "center_z_mm": 0.5}},
        mechanics_metadata={},
        provenance={"node_correspondence": "provable"},
        tetrahedra_node_ids=np.asarray(tetrahedra, dtype=np.int64),
    )
    result = NewtonResult(
        rest_vertices=vertices,
        deformed_vertices=vertices + vbd_displacement,
        tetrahedra=tetrahedra,
        steps=1,
    )
    return prepared, reference, result


def test_comparison_reports_full_field_and_geometry_descriptors() -> None:
    prepared, reference, result = _synthetic_states()

    comparison = compare_mechanics_states(reference, prepared, result)

    assert comparison["full_field"]["displacement_rms_error_mm"] > 0.0
    assert comparison["full_field"]["displacement_max_error_mm"] == pytest.approx(0.02)
    assert comparison["geometry_relevant"]["cavity_width"]["fea"]["cavity_width_change_mm"] == pytest.approx(0.1)
    assert comparison["geometry_relevant"]["cavity_width"]["vbd"]["cavity_width_change_mm"] == pytest.approx(0.12)
    assert comparison["geometry_relevant"]["surface_normal_change"]["fea_vbd_angular_error_deg"]["maximum"] >= 0.0
    assert "stress_field_agreement" in comparison["unsupported_quantities"]

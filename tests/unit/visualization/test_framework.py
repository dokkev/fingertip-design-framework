"""Synthetic contracts for dependency-light visualization transforms."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from visualization.data import (
    DisplacementField,
    MeshData,
    ObservationChain,
    ScientificFigureError,
)
from visualization.framework import (
    load_figure_spec,
)
from visualization.transforms import (
    CODTMVisualizationError,
    SelectedTransferState,
    deterministic_spatial_subsample,
    location_distance_matrix,
    mirror_side_swap,
    project_outward_displacement,
    select_indentation,
    shape_distance_matrix,
    signature_norm,
    symmetric_limits,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
def _chain(side: str, normal: tuple[float, float]) -> ObservationChain:
    return ObservationChain(
        side=side,
        point_ids=tuple(f"{side}:{index}" for index in range(5)),
        eta=np.linspace(0.0, 1.0, 5),
        undeformed_coordinates=np.stack(
            [
                np.full(5, -1.0 if side == "left" else 1.0),
                -np.linspace(0.0, 1.0, 5),
            ],
            axis=1,
        ),
        outward_normals=np.tile(normal, (5, 1)),
        mesh_id="synthetic",
        design_id="design",
        units="mm",
    )


def _selected_state(scale: float) -> SelectedTransferState:
    eta = np.linspace(0.0, 1.0, 5)
    return SelectedTransferState(
        design_id="design",
        mesh_id="synthetic",
        case_id=f"case-{scale}",
        xi=scale,
        delta_mm=1.0,
        reaction_force_n=1.0,
        contact_point_mm=(0.0, -1.0),
        indentation_direction=(0.0, 1.0),
        values_by_side={"left": scale * eta, "right": -scale * eta},
        eta_by_side={"left": eta, "right": eta},
        displacement_by_side={
            "left": np.stack([-scale * eta, eta * 0.0], axis=1),
            "right": np.stack([scale * eta, eta * 0.0], axis=1),
        },
        quantity="raw_displacement",
        units="mm",
        normalization="none",
        selection_metadata={"selection": "exact"},
    )


def test_json_compatible_yaml_spec_parses() -> None:
    spec = load_figure_spec(REPOSITORY_ROOT / "examples/transfer_map_comparison.yaml")
    assert spec.kind == "transfer_map_comparison"
    assert spec.quantity == "secant_gain"
    assert spec.contact_locations == (0.2, 0.35, 0.5, 0.65, 0.8)


def test_signed_figure_spec_rejects_nonzero_center() -> None:
    raw = json.loads(
        (REPOSITORY_ROOT / "examples/transfer_map_comparison.yaml").read_text()
    )
    raw["figure"]["color_scale"]["center"] = 1.0
    with pytest.raises(ScientificFigureError, match="zero-centered"):
        load_figure_spec(raw)


def test_mesh_data_preserves_explicit_topology() -> None:
    mesh = MeshData(
        node_ids=(9, 3, 17),
        node_coordinates=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        element_ids=(41,),
        element_connectivity=np.asarray([[9, 3, 17]]),
        spatial_dimension=2,
        mesh_id="mesh",
        design_id="design",
        units="mm",
    )
    assert mesh.element_connectivity.tolist() == [[9, 3, 17]]
    assert set(mesh.coordinate_by_node_id) == {3, 9, 17}


def test_mesh_data_rejects_unknown_connectivity_node() -> None:
    with pytest.raises(ScientificFigureError, match="unknown nodes"):
        MeshData(
            node_ids=(1, 2, 3),
            node_coordinates=np.zeros((3, 2)),
            element_ids=(1,),
            element_connectivity=np.asarray([[1, 2, 99]]),
            spatial_dimension=2,
            mesh_id="mesh",
            design_id="design",
            units="mm",
        )


def test_spatial_subsampling_is_mesh_order_independent() -> None:
    points = np.asarray(
        [(x, y) for x in np.linspace(-1.0, 1.0, 9) for y in (0.0, 1.0)]
    )
    first = points[deterministic_spatial_subsample(points, maximum_count=6)]
    permutation = np.asarray([7, 2, 15, 0, 9, 13, 5, 17, 1, 11, 4, 16, 8, 3, 14, 6, 10, 12])
    permuted = points[permutation]
    second = permuted[
        deterministic_spatial_subsample(permuted, maximum_count=6)
    ]
    assert np.array_equal(first, second)


def test_outward_projection_is_side_order_independent() -> None:
    left = _chain("left", (-1.0, 0.0))
    right = _chain("right", (1.0, 0.0))
    point_ids = right.point_ids + left.point_ids
    vectors = np.asarray([(2.0, 0.0)] * 5 + [(-3.0, 0.0)] * 5)
    field = DisplacementField(
        point_ids=point_ids,
        nodal_displacement=vectors,
        case_id="case",
        step=1,
        mesh_id="synthetic",
        design_id="design",
        represented_configuration="reference samples",
        validity_mask=np.ones(10, dtype=bool),
        units="mm",
        location_kind="observation_chain_sample",
    )
    projected = project_outward_displacement(
        field, {"right": right, "left": left}
    )
    assert projected["right"] == pytest.approx(np.full(5, 2.0))
    assert projected["left"] == pytest.approx(np.full(5, 3.0))


def test_eta_order_does_not_change_distance() -> None:
    eta = np.stack([np.linspace(0.0, 1.0, 8)] * 2)
    values = np.asarray(
        [[eta[0], eta[1] ** 2], [2.0 * eta[0], -eta[1]]]
    )
    expected = location_distance_matrix(values, eta)
    assert np.allclose(
        location_distance_matrix(values[..., ::-1], eta[..., ::-1]), expected
    )


def test_outward_projection_reproduces_affine_displacement() -> None:
    left = _chain("left", (-1.0, 0.0))
    right = _chain("right", (1.0, 0.0))
    coordinates = np.concatenate(
        [left.undeformed_coordinates, right.undeformed_coordinates]
    )
    # u=[2x+1, -3y] is evaluated analytically at actual sample coordinates.
    vectors = np.stack(
        [2.0 * coordinates[:, 0] + 1.0, -3.0 * coordinates[:, 1]], axis=1
    )
    field = DisplacementField(
        point_ids=left.point_ids + right.point_ids,
        nodal_displacement=vectors,
        case_id="affine",
        step=1,
        mesh_id="synthetic",
        design_id="design",
        represented_configuration="reference samples",
        validity_mask=np.ones(10, dtype=bool),
        units="mm",
        location_kind="observation_chain_sample",
    )
    result = project_outward_displacement(
        field, {"left": left, "right": right}
    )
    assert result["left"] == pytest.approx(np.ones(5))
    assert result["right"] == pytest.approx(np.full(5, 3.0))


def test_mirror_side_swap_preserves_eta_sample_order() -> None:
    result = mirror_side_swap(
        {"left": np.asarray([1.0, 2.0]), "right": np.asarray([8.0, 9.0])}
    )
    assert result["left"].tolist() == [8.0, 9.0]
    assert result["right"].tolist() == [1.0, 2.0]



def test_exact_indentation_selection() -> None:
    result = select_indentation(
        [0.5, 1.0], np.asarray([[2.0], [4.0]]), [True, True], 1.0
    )
    assert result.exact and result.values == pytest.approx([4.0])


def test_bracketed_interpolation_is_affine() -> None:
    result = select_indentation(
        [0.5, 1.0], np.asarray([[2.0], [4.0]]), [True, True], 0.75
    )
    assert not result.exact
    assert result.values == pytest.approx([3.0])


def test_extrapolation_is_rejected() -> None:
    with pytest.raises(CODTMVisualizationError, match="extrapolation"):
        select_indentation(
            [0.5, 1.0], np.asarray([[2.0], [4.0]]), [True, True], 1.5
        )


def test_invalid_bracket_is_rejected() -> None:
    with pytest.raises(CODTMVisualizationError, match="invalid step"):
        select_indentation(
            [0.5, 1.0], np.asarray([[2.0], [4.0]]), [True, False], 0.75
        )


def test_raw_distance_is_symmetric_with_exact_zero_diagonal() -> None:
    eta = np.stack([np.linspace(0.0, 1.0, 6)] * 2)
    values = np.arange(3 * 2 * 6, dtype=float).reshape(3, 2, 6)
    matrix = location_distance_matrix(values, eta)
    assert np.array_equal(matrix, matrix.T)
    assert np.array_equal(np.diag(matrix), np.zeros(3))


def test_side_integration_excludes_display_gap() -> None:
    eta = np.stack([np.linspace(0.0, 1.0, 101)] * 2)
    assert signature_norm(np.ones((2, 101)), eta) == pytest.approx(np.sqrt(2.0))


def test_shape_normalization_rejects_zero_norm() -> None:
    eta = np.stack([np.linspace(0.0, 1.0, 5)] * 2)
    with pytest.raises(CODTMVisualizationError, match="zero norm"):
        shape_distance_matrix(np.zeros((2, 2, 5)), eta)


def test_common_color_scale_uses_all_compared_states() -> None:
    first = _selected_state(0.2)
    second = _selected_state(0.8)
    assert symmetric_limits({"a": [first], "b": [second]}) == (-0.8, 0.8)


def test_vector_subsampling_is_deterministic() -> None:
    points = np.stack([np.linspace(-1.0, 1.0, 30), np.zeros(30)], axis=1)
    assert np.array_equal(
        deterministic_spatial_subsample(points, maximum_count=7),
        deterministic_spatial_subsample(points, maximum_count=7),
    )


def test_displacement_vectors_are_not_normalized() -> None:
    state = _selected_state(0.8)
    magnitudes = np.linalg.norm(state.displacement_by_side["right"], axis=1)
    assert magnitudes[-1] == pytest.approx(0.8)
    assert not np.allclose(magnitudes[1:], 1.0)

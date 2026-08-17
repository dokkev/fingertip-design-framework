"""Dependency-light contracts for the thin visualization API."""

from __future__ import annotations

from types import SimpleNamespace
import subprocess
import sys

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.quiver import Quiver
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

import visualization
from mesh import PadMesh
from model import Fingertip, FingertipParameters, LED, OpticalMaterial
from model.fingertip_model import FingertipModel
from optics import TraceSettings, trace
from visualization import (
    plot_camera,
    plot_case,
    plot_displacement,
    plot_fingertip,
    plot_mesh,
    plot_transport,
)


def _square_mesh() -> PadMesh:
    node_ids = np.asarray([10, 20, 30, 40], dtype=np.int64)
    coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )
    return PadMesh.from_arrays(
        node_ids=node_ids,
        reference_coordinates_mm=coordinates,
        element_connectivity_node_ids=np.asarray(
            [[10, 20, 30], [10, 30, 40]], dtype=np.int64
        ),
        boundary_edge_node_ids_by_tag={
            "bottom": np.asarray([[10, 20]], dtype=np.int64),
            "right": np.asarray([[20, 30]], dtype=np.int64),
            "top": np.asarray([[30, 40]], dtype=np.int64),
            "left": np.asarray([[40, 10]], dtype=np.int64),
        },
    )


def test_public_exports_are_only_plot_helpers() -> None:
    assert set(visualization.__all__) == {
        "plot_camera",
        "plot_case",
        "plot_displacement",
        "plot_fingertip",
        "plot_mesh",
        "plot_transport",
    }


def test_plot_fingertip_uses_only_the_public_facade() -> None:
    figure, axis = plt.subplots()
    plot_fingertip(Fingertip(FingertipParameters()), ax=axis)
    figure.canvas.draw()
    labels = {artist.get_label() for artist in (*axis.patches, *axis.lines)}
    assert "Silicone pad" in labels
    assert "LED" in labels
    assert any(isinstance(collection, PathCollection) for collection in axis.collections)
    assert axis.get_aspect() == 1.0
    with pytest.raises(TypeError, match="Fingertip"):
        plot_fingertip(FingertipModel(FingertipParameters()), ax=axis)  # type: ignore[arg-type]
    plt.close(figure)


def test_plot_mesh_draws_t3_connectivity_without_gmsh() -> None:
    mesh = _square_mesh()
    figure, axis = plt.subplots()
    assert plot_mesh(mesh, ax=axis, show_nodes=True) is axis
    assert len(axis.lines) >= 3
    assert any(isinstance(collection, PathCollection) for collection in axis.collections)
    assert axis.get_aspect() == 1.0
    plt.close(figure)


def test_plot_displacement_preserves_mesh_and_draws_magnitude_and_vectors() -> None:
    mesh = _square_mesh()
    before = mesh.coordinates.copy()
    displacement = np.asarray(
        [[0.0, 0.0], [0.10, 0.0], [0.10, 0.20], [0.0, 0.20]],
        dtype=float,
    )
    figure, axis = plt.subplots()
    plot_displacement(
        mesh,
        displacement,
        ax=axis,
        contact_point=(0.5, 0.0),
        indentation_direction=(0.0, 1.0),
    )
    assert np.array_equal(mesh.coordinates, before)
    assert any(isinstance(collection, Quiver) for collection in axis.collections)
    assert len(axis.collections) >= 2
    plt.close(figure)


@pytest.mark.parametrize(
    "bad, message",
    [
        (np.zeros((3, 2)), "same shape"),
        (np.asarray([[0.0, 0.0], [np.nan, 0.0], [0.0, 0.0], [0.0, 0.0]]), "finite"),
    ],
)
def test_plot_displacement_rejects_invalid_fields(bad, message) -> None:
    with pytest.raises(ValueError, match=message):
        plot_displacement(_square_mesh(), bad)
    with pytest.raises(ValueError, match="deformation_scale"):
        plot_displacement(_square_mesh(), np.zeros((4, 2)), deformation_scale=-1.0)
    with pytest.raises(ValueError, match="deformation_scale"):
        plot_displacement(_square_mesh(), np.zeros((4, 2)), deformation_scale=0.0)
    with pytest.raises(ValueError, match="arrow_scale"):
        plot_displacement(_square_mesh(), np.ones((4, 2)), arrow_scale=0.0)


def test_plot_transport_keeps_raw_analytic_density_unchanged() -> None:
    result = trace(
        Fingertip(
            FingertipParameters(),
            led=LED(emission_half_angle_deg=60.0),
        ),
        settings=TraceSettings(
            ray_count=17,
            grid_width=24,
            grid_height=24,
            maximum_segment_count=2000,
        ),
    )
    raw = result.density.copy()
    figure, axis = plt.subplots()
    assert plot_transport(result, ax=axis) is axis
    assert np.array_equal(result.density, raw)
    plt.close(figure)


def test_plot_case_composes_mechanics_pose_contact_and_p2() -> None:
    mesh = _square_mesh()
    displacement = np.zeros((4, 2), dtype=float)
    case = SimpleNamespace(
        parameters=FingertipParameters(),
        led=LED(),
        optical=OpticalMaterial(),
        fea=SimpleNamespace(
            mesh=mesh,
            displacement=displacement,
            deformed_mesh=mesh,
        ),
        indenter_pose=SimpleNamespace(
            carrier_geometry=Polygon(
                [(0.25, -0.4), (0.75, -0.4), (0.75, -0.1), (0.25, -0.1)]
            ),
            contact_patch=LineString([(0.4, 0.0), (0.6, 0.0)]),
        ),
        optics=SimpleNamespace(
            field=np.ones((3, 2), dtype=float),
            field_axes=(np.arange(4, dtype=float), np.arange(3, dtype=float)),
        ),
        raytrace=SimpleNamespace(
            escape_positions_mm=np.asarray([[0.5, 1.0, 0.0]]),
            escape_directions=np.asarray([[0.0, 1.0, 0.0]]),
        ),
    )

    figure = plot_case(case)
    figure.canvas.draw()
    assert len(figure.axes) >= 2
    plt.close(figure)


def test_plot_camera_does_not_import_mitsuba_or_mutate_rgb() -> None:
    rgb = np.asarray(
        [[[0.0, 0.2, 0.4], [0.5, 0.7, 1.0]], [[1.0, 0.4, 0.1], [0.2, 0.0, 0.8]]],
        dtype=float,
    )
    result = SimpleNamespace(linear_rgb=rgb.copy())
    before = result.linear_rgb.copy()
    figure, axis = plt.subplots()
    assert plot_camera(result, ax=axis, gamma=2.0) is axis
    assert np.array_equal(result.linear_rgb, before)
    with pytest.raises(ValueError, match="gamma"):
        plot_camera(result, ax=axis, gamma=0.0)
    plt.close(figure)


def test_import_visualization_is_optional_dependency_light() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys; import visualization; "
                "blocked = ('gmsh', 'KratosMultiphysics', 'mitsuba'); "
                "assert not any(name == item or name.startswith(item + '.') "
                "for name in sys.modules for item in blocked)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

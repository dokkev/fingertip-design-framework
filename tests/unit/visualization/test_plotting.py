"""Dependency-light contracts for the thin visualization API."""

from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.collections import QuadMesh
from matplotlib.colors import PowerNorm
from matplotlib.image import AxesImage
from matplotlib.quiver import Quiver
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString, Polygon

import visualization
import visualization.case as visualization_case
from mesh import PadMesh
from model import Fingertip, FingertipParameters, LED, OpticalMaterial
from model.fingertip_model import FingertipModel
from optics import TraceSettings, trace
from optics.transport3d import Transport3DSettings
from visualization import (
    plot_camera,
    plot_case_comparison,
    plot_fea,
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
        "plot_case_comparison",
        "plot_fea",
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


def test_plot_fea_preserves_mesh_and_draws_magnitude_and_vectors() -> None:
    mesh = _square_mesh()
    before = mesh.coordinates.copy()
    displacement = np.asarray(
        [[0.0, 0.0], [0.10, 0.0], [0.10, 0.20], [0.0, 0.20]],
        dtype=float,
    )
    figure, axis = plt.subplots()
    plot_fea(
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
def test_plot_fea_rejects_invalid_fields(bad, message) -> None:
    with pytest.raises(ValueError, match=message):
        plot_fea(_square_mesh(), bad)
    with pytest.raises(ValueError, match="deformation_scale"):
        plot_fea(_square_mesh(), np.zeros((4, 2)), deformation_scale=-1.0)
    with pytest.raises(ValueError, match="deformation_scale"):
        plot_fea(_square_mesh(), np.zeros((4, 2)), deformation_scale=0.0)
    with pytest.raises(ValueError, match="arrow_scale"):
        plot_fea(_square_mesh(), np.ones((4, 2)), arrow_scale=0.0)


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


def test_plot_case_comparison_builds_unloaded_loaded_2x2_without_execution() -> None:
    mesh = _square_mesh()
    displacement = np.asarray(
        [[0.0, 0.0], [0.10, 0.0], [0.10, 0.20], [0.0, 0.20]],
        dtype=float,
    )
    unloaded_field = np.asarray(
        [[1.0e-3, 2.0e-2, 1.0e-1], [0.0, 3.0e-3, 2.0e-1]],
        dtype=float,
    )
    loaded_field = np.asarray(
        [[2.0e-3, 4.0e-2, 2.0e-1], [0.0, 6.0e-3, 4.0e-1]],
        dtype=float,
    )
    pose = SimpleNamespace(
        carrier_geometry=Polygon(
            [(0.25, -0.4), (0.75, -0.4), (0.75, -0.1), (0.25, -0.1)]
        ),
        contact_patch=MultiLineString(
            [[(0.4, 0.0), (0.5, 0.0)], [(0.5, 0.0), (0.6, 0.0)]]
        ),
    )

    loaded_domain_mask = np.ones_like(loaded_field, dtype=bool)
    loaded_domain_mask[1, 2] = False

    def raw(field: np.ndarray, mask: np.ndarray | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            projected_x_edges_mm=np.arange(4, dtype=float),
            projected_y_edges_mm=np.arange(3, dtype=float),
            projected_weighted_path_density=field,
            projected_optical_mask=(
                np.ones_like(field, dtype=bool) if mask is None else mask
            ),
            escape_positions_mm=np.asarray([[0.5, 1.0, 0.0]]),
            escape_directions=np.asarray([[0.0, 1.0, 0.0]]),
            escape_weights=np.asarray([1.0]),
        )

    case = SimpleNamespace(
        fingertip=Fingertip(
            FingertipParameters(),
            led=LED(),
            optical=OpticalMaterial(),
        ),
        fea=SimpleNamespace(
            result=SimpleNamespace(
                mesh=mesh,
                displacement=displacement,
                deformed_mesh=mesh.deformed(displacement),
                reference_mesh=mesh,
                indenter_pose=pose,
                element_von_mises_stress_mpa={0: 0.25, 1: 0.75},
            ),
        ),
        raytracing=SimpleNamespace(
            raw=raw(loaded_field, loaded_domain_mask),
        ),
    )

    case.fea.solve = lambda *_args, **_kwargs: pytest.fail("plotting must not solve FEA")
    unloaded_before = unloaded_field.copy()
    loaded_before = loaded_field.copy()
    figure = plot_case_comparison(case, raw(unloaded_field), unloaded_pose=pose)
    figure.canvas.draw()
    panel_axes = [axis for axis in figure.axes if axis.get_title()]
    assert len(panel_axes) == 4
    assert {axis.get_title() for axis in panel_axes} == {
        "FEA — unloaded reference (zero stress)",
        "FEA — loaded",
        "PLANAR_2D OptiX — unloaded",
        "PLANAR_2D OptiX — loaded",
    }
    optical_axes = [axis for axis in panel_axes if "OptiX" in axis.get_title()]
    optical_collections = [
        image
        for axis in optical_axes
        for image in axis.collections
        if isinstance(image, QuadMesh)
    ]
    assert len(optical_collections) == 2
    assert optical_collections[0].norm is optical_collections[1].norm
    assert isinstance(optical_collections[0].norm, PowerNorm)
    assert optical_collections[0].norm.gamma == pytest.approx(0.45)
    assert optical_collections[0].norm.vmin == pytest.approx(0.0)
    assert optical_collections[0].norm.vmax < 4.0e-1
    assert any(np.ma.getmaskarray(collection.get_array()).any() for collection in optical_collections)
    loaded_display_mask = np.ma.getmaskarray(optical_collections[1].get_array()).reshape(-1)
    assert not loaded_display_mask[0]
    assert loaded_display_mask[-1]
    assert not any(isinstance(collection, Quiver) for axis in optical_axes for collection in axis.collections)
    assert any(
        axis.get_ylabel() == "Weighted optical path density"
        for axis in figure.axes
    )
    fea_axes = [axis for axis in panel_axes if axis.get_title().startswith("FEA")]
    stress_collections = [
        collection
        for axis in fea_axes
        for collection in axis.collections
        if isinstance(collection, PolyCollection)
    ]
    assert any(np.allclose(collection.get_array(), 0.0) for collection in stress_collections)
    assert any(np.max(collection.get_array()) == pytest.approx(0.75) for collection in stress_collections)
    assert np.array_equal(unloaded_field, unloaded_before)
    assert np.array_equal(loaded_field, loaded_before)
    assert np.array_equal(case.raytracing.raw.projected_weighted_path_density, loaded_field)

    debug_figure = plot_case_comparison(
        case,
        raw(unloaded_field),
        unloaded_pose=pose,
        show_exits=True,
    )
    debug_optical_axes = [
        axis for axis in debug_figure.axes if "OptiX" in axis.get_title()
    ]
    assert any(
        isinstance(collection, Quiver)
        for axis in debug_optical_axes
        for collection in axis.collections
    )
    plt.close(debug_figure)
    plt.close(figure)


def test_case_display_transform_does_not_enter_evaluation_code() -> None:
    source = inspect.getsource(visualization_case)
    assert "evaluate(" not in source
    assert "optics.metrics" not in visualization_case.__dict__


def test_publication_example_uses_high_resolution_transport_defaults() -> None:
    source = (
        Path(__file__).resolve().parents[3] / "examples" / "view_case.py"
    ).read_text(encoding="utf-8")
    assert "ray_count=256" not in source
    assert "max_interactions=8" not in source
    assert "projected_grid_width=96" not in source
    assert "projected_grid_height=96" not in source

    settings = Transport3DSettings(
        mode="planar",
        retain_projected_segments=True,
    )
    assert settings.ray_count == 4096
    assert settings.max_interactions == 10
    assert settings.projected_grid_width == 240
    assert settings.projected_grid_height == 240


def test_display_smoothing_is_optional_and_does_not_mutate_raw_field() -> None:
    field = np.asarray(
        [[0.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    before = field.copy()
    smoothed = visualization_case._smooth_display_field(
        field,
        np.ones_like(field, dtype=bool),
        radius_cells=1,
    )
    assert np.array_equal(field, before)
    assert smoothed.shape == field.shape


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

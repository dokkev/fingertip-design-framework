from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
import pytest

pytest.importorskip("gmsh")

from mesh import FingertipVolumeState, volume_mesh_settings_for_tier
from model import Fingertip
from visualization import plot_volume_mesh, plot_volume_state
from visualization.volume import draw_volume_mesh


@pytest.fixture(scope="module")
def nominal_volume_state():
    tip = Fingertip()
    volume_mesh = tip.volume_mesh(volume_mesh_settings_for_tier("search"))
    return volume_mesh, FingertipVolumeState.reference(volume_mesh)


def test_plot_volume_mesh_renders_selected_semantic_shell_and_rejects_unknown_tag(
    nominal_volume_state,
) -> None:
    volume_mesh, _ = nominal_volume_state
    figure = plt.figure()
    axis = plot_volume_mesh(
        volume_mesh,
        surface_tags=("outer_compliant_arc",),
        show_edges=True,
        show_nodes=True,
        ax=figure.add_subplot(111, projection="3d"),
    )
    figure.canvas.draw()

    assert axis.name == "3d"
    assert any(isinstance(collection, Poly3DCollection) for collection in axis.collections)
    assert axis.get_xlabel() == "x [mm]"
    assert axis.get_ylabel() == "y [mm]"
    assert axis.get_zlabel() == "z [mm]"
    assert len(axis.collections) >= 2

    with pytest.raises(KeyError, match="unknown volume surface tag"):
        plot_volume_mesh(volume_mesh, surface_tags=("does_not_exist",))
    plt.close(figure)


def test_draw_volume_mesh_preserves_existing_axes_policy(nominal_volume_state) -> None:
    volume_mesh, _ = nominal_volume_state
    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-2.0, 2.0)
    axis.set_zlim(-3.0, 3.0)
    axis.view_init(elev=7.0, azim=11.0)
    axis.set_xlabel("existing x")
    draw_volume_mesh(axis, volume_mesh, surface_tags=("outer_compliant_arc",))

    np.testing.assert_allclose(axis.get_xlim(), (-1.0, 1.0))
    np.testing.assert_allclose(axis.get_ylim(), (-2.0, 2.0))
    np.testing.assert_allclose(axis.get_zlim(), (-3.0, 3.0))
    assert axis.get_xlabel() == "existing x"
    assert axis.elev == pytest.approx(7.0)
    assert axis.azim == pytest.approx(11.0)
    plt.close(figure)


def test_plot_volume_state_displacement_uses_neutral_field_and_shared_norm(
    nominal_volume_state,
) -> None:
    volume_mesh, reference_state = nominal_volume_state
    deformed = reference_state.reference_coordinates_mm.copy()
    deformed[:, 0] += 0.025
    state = FingertipVolumeState.from_deformed_coordinates(volume_mesh, deformed)
    before = state.deformed_coordinates_mm.copy()
    norm = Normalize(vmin=0.0, vmax=0.1)

    figure = plt.figure()
    axis = plot_volume_state(
        state,
        norm=norm,
        show_reference=True,
        deformation_scale=2.0,
        highlight_vertex_indices=(0, 1),
        ax=figure.add_subplot(111, projection="3d"),
    )
    figure.canvas.draw()

    surface = next(
        collection
        for collection in axis.collections
        if isinstance(collection, Poly3DCollection)
    )
    assert surface.norm is norm
    np.testing.assert_allclose(surface.get_array(), 0.025)
    assert any(isinstance(collection, Line3DCollection) for collection in axis.collections)
    assert len(axis.figure.axes) == 2
    assert axis.figure.axes[1].get_ylabel() == "displacement magnitude [mm]"
    assert np.array_equal(state.deformed_coordinates_mm, before)
    plt.close(figure)


def test_plot_volume_state_can_defer_colorbar_for_shared_composition(nominal_volume_state) -> None:
    volume_mesh, state = nominal_volume_state
    figure = plt.figure()
    axis = plot_volume_state(
        state,
        colorbar=False,
        ax=figure.add_subplot(111, projection="3d"),
    )
    figure.canvas.draw()
    assert axis.name == "3d"
    assert len(figure.axes) == 1
    plt.close(figure)


def test_plot_volume_state_semantic_mode_and_backend_independence(nominal_volume_state) -> None:
    volume_mesh, reference_state = nominal_volume_state
    second_state = FingertipVolumeState.from_deformed_coordinates(
        volume_mesh,
        reference_state.deformed_coordinates_mm.copy(),
    )
    figure = plt.figure()
    first_axis = plot_volume_state(
        reference_state,
        field="semantic",
        surface_tags=("outer_compliant_arc", "void_left"),
        show_reference=False,
        elev=35.0,
        azim=-45.0,
        ax=figure.add_subplot(121, projection="3d"),
    )
    second_axis = plot_volume_state(
        second_state,
        field="semantic",
        surface_tags=("outer_compliant_arc", "void_left"),
        show_reference=False,
        ax=figure.add_subplot(122, projection="3d"),
    )
    figure.canvas.draw()

    first_surface = next(
        collection
        for collection in first_axis.collections
        if isinstance(collection, Poly3DCollection)
    )
    second_surface = next(
        collection
        for collection in second_axis.collections
        if isinstance(collection, Poly3DCollection)
    )
    assert len(first_surface.get_paths()) == len(second_surface.get_paths())
    np.testing.assert_allclose(first_axis.get_xlim(), second_axis.get_xlim())
    np.testing.assert_allclose(first_axis.get_ylim(), second_axis.get_ylim())
    np.testing.assert_allclose(first_axis.get_zlim(), second_axis.get_zlim())
    with pytest.raises(ValueError, match="field"):
        plot_volume_state(reference_state, field="solver_name")
    plt.close(figure)

"""Headless smoke coverage for the parameterized geometry figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import FancyArrowPatch, PathPatch

from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters
from visualization.geometry import plot_fingertip


def test_revised_csg_geometry_renders_as_distinct_material_and_void(tmp_path) -> None:
    parameters = FingertipParameters(void_width=1.0, void_height=2.0)
    model = FingertipModel(parameters)
    figure, axis = plt.subplots(figsize=(8.0, 6.0))
    plot_fingertip(
        model,
        ax=axis,
        show_void=True,
        show_interface=True,
        show_contact_boundaries=True,
    )
    figure.canvas.draw()

    polygon_patches = {
        patch.get_label(): patch
        for patch in axis.patches
        if isinstance(patch, PathPatch)
    }
    assert set(polygon_patches) == {
        "Silicone pad",
        "Rigid link / stem",
        "_nolegend_",
        "Void",
    }
    assert any(isinstance(patch, FancyArrowPatch) for patch in axis.patches)
    assert not axis.texts

    def path_bounds(patch: PathPatch) -> tuple[float, float, float, float]:
        vertices = np.asarray(patch.get_path().vertices)
        return (
            float(vertices[:, 0].min()),
            float(vertices[:, 1].min()),
            float(vertices[:, 0].max()),
            float(vertices[:, 1].max()),
        )

    assert path_bounds(polygon_patches["Silicone pad"]) == pytest.approx(
        model.pad_material_geometry.bounds
    )
    assert model.void_geometry is not None
    assert path_bounds(polygon_patches["Void"]) == pytest.approx(
        model.void_geometry.bounds
    )
    assert model.void_geometry.equals(
        model.cutout_geometry.difference(model.stem_geometry)
    )

    output = tmp_path / "fingertip_geometry.png"
    figure.savefig(output)
    plt.close(figure)
    assert output.stat().st_size > 0

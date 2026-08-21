"""Tests for explicit OBJ rigid-mesh asset conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mesh.io.obj import RigidMeshAssetError, load_obj, load_obj_directory, save_obj
from mesh.rigid.object import make_box_mesh, make_sphere_mesh


def test_save_and_load_obj_round_trip_preserves_rigid_mesh(tmp_path: Path) -> None:
    source = make_sphere_mesh(2.0, subdivisions=1)
    path = save_obj(source, tmp_path / "sphere.obj")

    loaded = load_obj(path, scale_mm_per_unit=1.0)

    assert loaded.name == "sphere"
    np.testing.assert_allclose(loaded.vertices_mm, source.vertices_mm)
    np.testing.assert_array_equal(loaded.faces, source.faces)
    assert loaded.name == "sphere"


def test_load_obj_applies_explicit_scale_and_triangulates_polygon(tmp_path: Path) -> None:
    path = tmp_path / "box.obj"
    path.write_text(
        "\n".join(
            (
                "v -1 -1 -1",
                "v 1 -1 -1",
                "v 1 1 -1",
                "v -1 1 -1",
                "v -1 -1 1",
                "v 1 -1 1",
                "v 1 1 1",
                "v -1 1 1",
                "f 1 4 3 2",
                "f 5 6 7 8",
                "f 1 2 6 5",
                "f 4 8 7 3",
                "f 1 5 8 4",
                "f 2 3 7 6",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_obj(path, scale_mm_per_unit=2.0)

    assert loaded.bounds_mm == ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))
    assert loaded.faces.shape == (12, 3)


def test_load_obj_directory_is_sorted_and_uses_stem_names(tmp_path: Path) -> None:
    save_obj(make_box_mesh(1.0, 1.0, 1.0), tmp_path / "b.obj")
    save_obj(make_box_mesh(2.0, 2.0, 2.0), tmp_path / "a.obj")

    assets = load_obj_directory(tmp_path, scale_mm_per_unit=1.0)

    assert list(assets) == ["a", "b"]
    assert assets["a"].bounds_mm == ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_load_obj_rejects_invalid_scale(tmp_path: Path, scale: float) -> None:
    path = save_obj(make_box_mesh(1.0, 1.0, 1.0), tmp_path / "box.obj")

    with pytest.raises(RigidMeshAssetError, match="scale_mm_per_unit"):
        load_obj(path, scale_mm_per_unit=scale)

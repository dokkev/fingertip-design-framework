"""Shared Gmsh initialization failures remain infrastructure failures."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import mesh.fingertip.geometry as surface_mesh_module
import mesh.volume.mesh as volume_mesh_module
from mesh.fingertip.contracts import mesh_settings_for_level
from mesh.fingertip.geometry import FingertipMeshingError, GmshDependencyError
from mesh.volume.contracts import volume_mesh_settings_for_tier
from mesh.volume.mesh import VolumeMeshDependencyError, VolumeMeshingError
from model import Fingertip


def _failed_gmsh() -> SimpleNamespace:
    def initialize(*_args, **_kwargs) -> None:
        raise OSError("shared Gmsh library is unavailable")

    return SimpleNamespace(initialize=initialize)


def _operationally_failed_gmsh() -> SimpleNamespace:
    def fail_add(*_args, **_kwargs) -> None:
        raise RuntimeError("candidate OCC construction failed")

    return SimpleNamespace(
        initialize=lambda *_args, **_kwargs: None,
        finalize=lambda: None,
        model=SimpleNamespace(add=fail_add),
    )


def test_volume_mesh_initialization_failure_is_infrastructure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(volume_mesh_module, "_import_gmsh", _failed_gmsh)

    with pytest.raises(VolumeMeshDependencyError, match="could not initialize"):
        volume_mesh_module.generate_volume_mesh(
            Fingertip().solid(),
            volume_mesh_settings_for_tier("search"),
        )


def test_surface_mesh_initialization_failure_is_infrastructure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(surface_mesh_module, "_import_gmsh", _failed_gmsh)

    with pytest.raises(GmshDependencyError, match="could not initialize"):
        surface_mesh_module.generate_fingertip_mesh(
            Fingertip().geometry,
            mesh_settings_for_level("medium"),
        )


def test_volume_mesh_operational_failure_is_candidate_meshing_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        volume_mesh_module,
        "_import_gmsh",
        _operationally_failed_gmsh,
    )

    with pytest.raises(VolumeMeshingError, match="candidate OCC"):
        volume_mesh_module.generate_volume_mesh(
            Fingertip().solid(),
            volume_mesh_settings_for_tier("search"),
        )


def test_surface_mesh_operational_failure_is_candidate_meshing_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        surface_mesh_module,
        "_import_gmsh",
        _operationally_failed_gmsh,
    )

    with pytest.raises(FingertipMeshingError, match="candidate OCC"):
        surface_mesh_module.generate_fingertip_mesh(
            Fingertip().geometry,
            mesh_settings_for_level("medium"),
        )

"""Procedural in-memory Mitsuba scene construction."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from model.fingertip import Fingertip
from optics.geometry.extrusion import _ExtrudedMesh
from optics.mitsuba.parameters import Camera, RenderSettings


class MitsubaSceneError(RuntimeError):
    """Raised when an in-memory Mitsuba scene cannot be constructed."""


def _camera_dict(mi: Any, camera: Camera) -> dict[str, Any]:
    to_world = mi.ScalarTransform4f.look_at(
        origin=list(camera.position_mm),
        target=list(camera.target_mm),
        up=list(camera.up),
    )
    sensor: dict[str, Any] = {
        "type": camera.projection,
        "to_world": to_world,
        "film": {
            "type": "hdrfilm",
            "width": camera.resolution_px[0],
            "height": camera.resolution_px[1],
            "pixel_format": "rgb",
            "component_format": "float32",
            "rfilter": {"type": "box"},
        },
    }
    if camera.projection == "orthographic":
        scale = float(camera.orthographic_scale_mm)
        sensor["to_world"] = to_world @ mi.ScalarTransform4f.scale(
            [scale, scale, 1.0]
        )
    else:
        sensor["fov"] = camera.fov_deg
    return sensor


def build_in_memory_mitsuba_scene(
    mi: Any,
    *,
    tip: Fingertip,
    extrusion: _ExtrudedMesh,
    vertices_mm: np.ndarray,
    camera: Camera,
    settings: RenderSettings,
    source_positions_mm: Sequence[tuple[float, float, float]] | None = None,
) -> Any:
    """Build one scene containing a procedural pad mesh and point emitters."""
    vertices = np.asarray(vertices_mm, dtype=np.float32)
    faces = np.asarray(extrusion.faces_3d, dtype=np.uint32)
    if vertices.shape != (2 * extrusion.node_count_2d, 3):
        raise MitsubaSceneError("extruded vertices have an unexpected shape")

    material = tip.optical
    mesh_properties = mi.Properties()
    mesh_properties["bsdf"] = mi.load_dict({"type": "null"})
    mesh_properties["interior"] = mi.load_dict(
        {
            "type": "homogeneous",
            "sigma_t": material.extinction_per_mm,
            "albedo": material.single_scattering_albedo,
            "phase": {"type": "hg", "g": material.anisotropy_g},
        }
    )
    mesh = mi.Mesh(
        "pad",
        vertex_count=len(vertices),
        face_count=len(faces),
        props=mesh_properties,
        has_vertex_normals=False,
        has_vertex_texcoords=False,
    )
    mesh_parameters = mi.traverse(mesh)
    mesh_parameters["vertex_positions"] = vertices.reshape(-1)
    mesh_parameters["faces"] = faces.reshape(-1)
    mesh_parameters.update()

    if source_positions_mm is None:
        source_positions = ((tip.led_source[0], tip.led_source[1], 0.0),)
    else:
        source_positions = tuple(source_positions_mm)
    if not source_positions:
        raise MitsubaSceneError("source_positions_mm must not be empty")
    sources: list[tuple[str, dict[str, Any]]] = []
    for index, position_mm in enumerate(source_positions):
        position = np.asarray(position_mm, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise MitsubaSceneError(
                "source_positions_mm must contain finite 3D positions"
            )
        position = position + settings.source_epsilon_mm * np.asarray(
            [0.0, -1.0, 0.0]
        )
        name = "led" if len(source_positions) == 1 else f"led_{index}"
        sources.append(
            (
                name,
                {
                    "type": "point",
                    "position": position.tolist(),
                },
            )
        )
    intensity = settings.point_emitter_scale * np.asarray(
        tip.led.emission_rgb,
        dtype=float,
    ) * tip.led.relative_radiant_power
    scene: dict[str, Any] = {
        "type": "scene",
        "integrator": {
            "type": "volpath",
            "max_depth": settings.max_depth,
        },
        "pad": mesh,
        "camera": _camera_dict(mi, camera),
    }
    for name, source in sources:
        source["intensity"] = {
            "type": "rgb",
            "value": intensity.tolist(),
        }
        scene[name] = source
    return mi.load_dict(scene)

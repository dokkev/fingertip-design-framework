"""Procedural in-memory Mitsuba scene construction."""

from __future__ import annotations

from typing import Any

import numpy as np

from model.fingertip_sensor_model import FingertipSensorModel
from optics.geometry.extrusion import ExtrudedOpticalMeshTemplate
from optics.mitsuba.parameters import MitsubaCameraParameters, MitsubaRenderSettings


class MitsubaSceneError(RuntimeError):
    """Raised when an in-memory Mitsuba scene cannot be constructed."""


def _camera_dict(mi: Any, camera: MitsubaCameraParameters) -> dict[str, Any]:
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
    sensor_model: FingertipSensorModel,
    extrusion: ExtrudedOpticalMeshTemplate,
    vertices_mm: np.ndarray,
    camera: MitsubaCameraParameters,
    settings: MitsubaRenderSettings,
) -> Any:
    """Build one scene containing a procedural pad mesh and point emitter."""
    vertices = np.asarray(vertices_mm, dtype=np.float32)
    faces = np.asarray(extrusion.faces_3d, dtype=np.uint32)
    if vertices.shape != (2 * extrusion.node_count_2d, 3):
        raise MitsubaSceneError("extruded vertices have an unexpected shape")

    material = sensor_model.optical_material
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

    source = np.asarray(sensor_model.led_source_position_3d, dtype=float)
    source += settings.source_epsilon_mm * np.asarray([0.0, -1.0, 0.0])
    intensity = settings.point_emitter_scale * np.asarray(
        sensor_model.led.emission_rgb,
        dtype=float,
    ) * sensor_model.led.relative_radiant_power
    return mi.load_dict(
        {
            "type": "scene",
            "integrator": {
                "type": "volpath",
                "max_depth": settings.max_depth,
            },
            "pad": mesh,
            "led": {
                "type": "point",
                "position": source.tolist(),
                "intensity": {
                    "type": "rgb",
                    "value": intensity.tolist(),
                },
            },
            "camera": _camera_dict(mi, camera),
        }
    )

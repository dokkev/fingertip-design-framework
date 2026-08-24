"""Small viewer helpers for validation and debugging scripts."""

from __future__ import annotations

from typing import Any


def make_reference_lines() -> tuple[Any, Any, Any]:
    """Return an X-Y floor grid and colored LUMO-coordinate axes."""
    import warp as wp

    grid_extent_m = 0.04
    grid_step_m = 0.005
    floor_z_m = -0.02

    starts: list[wp.vec3] = []
    ends: list[wp.vec3] = []
    colors: list[wp.vec3] = []
    grid_color = wp.vec3(0.25, 0.25, 0.28)

    grid_count = int(grid_extent_m / grid_step_m)
    for index in range(-grid_count, grid_count + 1):
        coordinate_m = index * grid_step_m

        starts.append(wp.vec3(-grid_extent_m, coordinate_m, floor_z_m))
        ends.append(wp.vec3(grid_extent_m, coordinate_m, floor_z_m))
        colors.append(grid_color)

        starts.append(wp.vec3(coordinate_m, -grid_extent_m, floor_z_m))
        ends.append(wp.vec3(coordinate_m, grid_extent_m, floor_z_m))
        colors.append(grid_color)

    axis_length_m = 0.025
    starts.extend(
        (
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
        )
    )
    ends.extend(
        (
            wp.vec3(axis_length_m, 0.0, 0.0),
            wp.vec3(0.0, axis_length_m, 0.0),
            wp.vec3(0.0, 0.0, axis_length_m),
        )
    )
    colors.extend(
        (
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            wp.vec3(0.0, 0.0, 1.0),
        )
    )

    return (
        wp.array(starts, dtype=wp.vec3),
        wp.array(ends, dtype=wp.vec3),
        wp.array(colors, dtype=wp.vec3),
    )


def configure_fingertip_camera(viewer: Any) -> None:
    """Configure Newton ViewerGL for the centimeter-scale fingertip scene."""
    import warp as wp

    viewer.set_camera(
        wp.vec3(0.08, -0.08, 0.05),
        -25.0,
        135.0,
    )
    viewer.camera.fov = 25.0
    viewer.camera_speed = 0.03

    # ViewerGL does not expose mouse sensitivities as public settings.
    # This helper is limited to validation/debug viewers, so keep the local
    # tuning explicit rather than adding it to production runtime code.
    viewer.gui._camera_orbit_sensitivity = 0.03
    viewer.gui._camera_dolly_scroll_sensitivity = 0.03
    viewer.gui._camera_dolly_drag_sensitivity = 0.003


__all__ = ["configure_fingertip_camera", "make_reference_lines"]

"""Inspect one Newton VBD rigid-mesh contact indentation interactively.

This example uses the production 3D contact path, not the legacy 2D FEM
indenter.  The default ``ViewerGL`` window shows the tetrahedral fingertip,
kinematic triangle-mesh indenter, and the contacts reported by Newton.

Examples::

    conda run -n lit python examples/view_newton_contact.py
    conda run -n lit python examples/view_newton_contact.py --no-viewer
    conda run -n lit python examples/view_newton_contact.py --usd output/contact.usda
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bootstrap import ensure_repository_root

ensure_repository_root()

import numpy as np
import warp as wp

from contact import (
    FirstContactSettings,
    canonical_sphere_alignment,
    find_first_contact,
    intersects,
    make_outer_compliant_surface,
)
from mechanics3d import (
    IndentationSettings,
    Mechanics3DSettings,
    RigidIndenter3D,
    prepare_fingertip_mechanics_mesh,
)
from mechanics3d.backends.newton_vbd import solve_newton_vbd_indentation
from mesh import (
    make_distal_phalanx_mesh,
    make_sphere_mesh,
)
from mesh.volume3d import generate_volume_mesh
from mesh.volume_types import volume_mesh_settings_for_tier
from model import Fingertip, FingertipParameters
from util.newton_viewer import (
    close_newton_viewer,
    frame_newton_viewer,
    keep_newton_viewer_open,
    make_newton_viewer,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--travel",
        type=float,
        default=0.6,
        help="post-contact sphere travel in mm",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="sphere radius in mm",
    )
    parser.add_argument(
        "--sphere-subdivisions",
        type=int,
        default=1,
        help="icosphere subdivision level",
    )
    parser.add_argument(
        "--initial-gap",
        type=float,
        default=0.25,
        help="free-space placement gap used before geometric first-contact search",
    )
    parser.add_argument("--load-steps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="run the contact solve and print diagnostics without opening a window",
    )
    parser.add_argument(
        "--usd",
        type=Path,
        help="write accepted Newton states to this USD file instead of opening ViewerGL",
    )
    return parser


def _build_scene():
    fingertip = Fingertip(
        FingertipParameters(
            void_width=1.0,
            void_height=0.0,
            poisson_ratio=0.49,
        )
    )
    volume_mesh = generate_volume_mesh(
        fingertip.solid(),
        volume_mesh_settings_for_tier("search"),
    )
    prepared = prepare_fingertip_mechanics_mesh(volume_mesh)
    return fingertip, volume_mesh, prepared


def _rigid_vertices_world_mm(object_mesh, pose) -> np.ndarray:
    """Return rigid-object vertices in world millimetres for one pose."""

    qx, qy, qz, qw = pose.quaternion_xyzw
    rotation = np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )
    translation = np.asarray(pose.translation_mm, dtype=float)
    return np.asarray(object_mesh.vertices_mm, dtype=float) @ rotation.T + translation


def _scene_bounds_m(
    prepared,
    object_mesh,
    poses,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute fingertip plus swept indenter bounds in Newton metres."""

    fingertip_vertices_mm = np.asarray(prepared.tet_mesh.vertices, dtype=float)
    rigid_vertices_mm = np.concatenate(
        [
            *(_rigid_vertices_world_mm(object_mesh, pose) for pose in poses),
        ],
        axis=0,
    )
    scene_vertices_m = np.concatenate(
        [fingertip_vertices_mm, rigid_vertices_mm],
        axis=0,
    ) * 1.0e-3
    return scene_vertices_m.min(axis=0), scene_vertices_m.max(axis=0)


def main() -> int:
    args = _parser().parse_args()
    if not wp.is_device_available(args.device):
        raise RuntimeError(f"CUDA device {args.device!r} is not available")

    fingertip, volume_mesh, prepared = _build_scene()
    object_mesh = make_sphere_mesh(
        args.radius,
        subdivisions=args.sphere_subdivisions,
    )
    alignment = canonical_sphere_alignment(
        fingertip.geometry,
        object_mesh,
        initial_gap_mm=args.initial_gap,
    )
    contact_surface = make_outer_compliant_surface(volume_mesh.solid)
    first_contact_settings = FirstContactSettings(
        coarse_step_mm=0.25,
        tolerance_mm=1.0e-3,
        spawn_clearance_mm=0.05,
        max_travel_mm=20.0,
    )
    if intersects(contact_surface, object_mesh, alignment.nominal_pose):
        raise RuntimeError("canonical sphere nominal pose is not collision-free")
    first_contact = find_first_contact(
        contact_surface,
        object_mesh,
        alignment.nominal_pose,
        alignment.approach_direction,
        first_contact_settings,
    )
    indenter = RigidIndenter3D(
        object_mesh,
        alignment.nominal_pose,
        alignment.approach_direction,
    )
    final_pose = first_contact.pose_at_post_contact_travel(args.travel)
    scene_bounds_m = _scene_bounds_m(
        prepared,
        object_mesh,
        (first_contact.spawn_pose, final_pose),
    )
    viewer = make_newton_viewer(
        no_viewer=args.no_viewer,
        usd_path=args.usd,
        num_frames=args.load_steps,
    )
    visual_carrier_mesh = (
        make_distal_phalanx_mesh(volume_mesh.solid) if viewer is not None else None
    )
    try:
        result = solve_newton_vbd_indentation(
            prepared,
            indenter,
            Mechanics3DSettings(
                device=args.device,
                gravity=0.0,
                dt=1.0e-3,
                steps=1,
                iterations=args.iterations,
                fixed_vertex_indices=prepared.support_vertex_indices,
            ),
            IndentationSettings(
                travel_mm=args.travel,
                load_steps=args.load_steps,
                soft_contact_margin_mm=0.02,
                soft_contact_ke=1.0e3,
                soft_contact_kd=10.0,
            ),
            viewer=viewer,
            visual_carrier_mesh=visual_carrier_mesh,
            first_contact=first_contact,
        )
        if viewer is not None and args.usd is None:
            frame_newton_viewer(
                viewer,
                *scene_bounds_m,
                view_direction=(0.0, -1.0, 0.4),
                padding=1.4,
            )
            keep_newton_viewer_open(viewer)
    finally:
        close_newton_viewer(viewer)

    print(f"object: {object_mesh.name} (radius_mm={alignment.radius_mm:g})")
    print(f"target_point_mm: {alignment.target_point_mm}")
    print(f"approach_direction: {alignment.approach_direction}")
    print(f"reference_pose_collision_free: {not intersects(contact_surface, object_mesh, alignment.nominal_pose)}")
    print(f"first_contact_travel_mm: {first_contact.travel_to_contact_mm:.6g}")
    print(f"first_contact_bracket_width_mm: {first_contact.bracket_width_mm:.6g}")
    print(f"first_contact_tolerance_mm: {first_contact_settings.tolerance_mm:g}")
    print(f"spawn_clearance_mm: {first_contact.spawn_clearance_mm:g}")
    print(f"spawn_pose_collision_free: {not intersects(contact_surface, object_mesh, first_contact.spawn_pose)}")
    print(f"post_contact_travel_mm: {args.travel:g}")
    print(f"full_surface_contact: {result.diagnostics['full_surface_contact']}")
    print(f"soft_contacts: {result.diagnostics['max_soft_contact_count']}")
    print(f"rigid_contacts: {result.diagnostics['max_rigid_contact_count']}")
    print(f"max_displacement_mm: {result.diagnostics['max_displacement_mm']:.6g}")
    if args.usd is not None:
        print(f"usd: {args.usd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Inspect one Newton VBD rigid-mesh contact indentation interactively.

This example uses the production 3D contact path, not the legacy 2D FEM
indenter.  The default ``ViewerGL`` window shows the tetrahedral fingertip,
kinematic triangle-mesh indenter, and the contacts reported by Newton.

Examples::

    conda run -n lit python examples/view_newton_contact.py
    conda run -n lit python examples/view_newton_contact.py --object cylinder
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

from mechanics3d import (
    IndentationSettings,
    Mechanics3DSettings,
    RigidIndenter3D,
    RigidPose3D,
    prepare_fingertip_mechanics_mesh,
)
from mechanics3d.backends.newton_vbd import solve_newton_vbd_indentation
from mesh import make_box_mesh, make_cylinder_mesh, make_sphere_mesh
from mesh.volume3d import generate_volume_mesh
from mesh.volume_types import volume_mesh_settings_for_tier
from model import Fingertip, FingertipParameters
from util.newton_viewer import (
    close_newton_viewer,
    keep_newton_viewer_open,
    make_newton_viewer,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object",
        choices=("sphere", "cylinder", "box"),
        default="sphere",
        help="triangle-mesh rigid object used for contact",
    )
    parser.add_argument("--travel", type=float, default=0.6, help="indentation travel in mm")
    parser.add_argument("--load-steps", type=int, default=8)
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


def _make_object(kind: str):
    if kind == "sphere":
        return make_sphere_mesh(2.0, subdivisions=1)
    if kind == "cylinder":
        return make_cylinder_mesh(1.8, 3.0, radial_segments=16)
    return make_box_mesh(3.0, 3.0, 3.0)


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
    return volume_mesh, prepared


def _make_indenter(prepared, object_mesh):
    surface_candidates = np.unique(prepared.surface_triangles["outer_compliant_arc"])
    local_surface = surface_candidates[
        np.abs(prepared.tet_mesh.vertices[surface_candidates, 0] - 10.0) < 1.0
    ]
    contact_y_mm = float(prepared.tet_mesh.vertices[local_surface, 1].max())
    object_top_mm = float(object_mesh.vertices_mm[:, 1].max())
    return RigidIndenter3D(
        object_mesh,
        RigidPose3D(
            (10.0, contact_y_mm + object_top_mm + 0.5, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        (0.0, -1.0, 0.0),
    )


def main() -> int:
    args = _parser().parse_args()
    if not wp.is_device_available(args.device):
        raise RuntimeError(f"CUDA device {args.device!r} is not available")

    _, prepared = _build_scene()
    object_mesh = _make_object(args.object)
    indenter = _make_indenter(prepared, object_mesh)
    viewer = make_newton_viewer(
        no_viewer=args.no_viewer,
        usd_path=args.usd,
        num_frames=args.load_steps,
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
                iterations=5,
                fixed_vertex_indices=prepared.support_vertex_indices,
            ),
            IndentationSettings(
                travel_mm=args.travel,
                load_steps=args.load_steps,
                soft_contact_margin_mm=0.02,
                soft_contact_ke=1.0e3,
                soft_contact_kd=10.0,
                rigid_body_particle_contact_buffer_size=8192,
            ),
            viewer=viewer,
        )
        if viewer is not None and args.usd is None:
            keep_newton_viewer_open(viewer)
    finally:
        close_newton_viewer(viewer)

    print(f"object: {object_mesh.name}")
    print(f"full_surface_contact: {result.diagnostics['full_surface_contact']}")
    print(f"soft_contacts: {result.diagnostics['max_soft_contact_count']}")
    print(f"rigid_contacts: {result.diagnostics['max_rigid_contact_count']}")
    print(f"max_displacement_mm: {result.diagnostics['max_displacement_mm']:.6g}")
    if args.usd is not None:
        print(f"usd: {args.usd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

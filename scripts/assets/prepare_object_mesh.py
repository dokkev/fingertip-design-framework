#!/usr/bin/env python3
"""Prepare deterministic parametric rigid-object OBJ assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from mesh.io.obj import save_obj
from mesh.rigid.object import make_sphere_mesh


def _default_sphere_path(radius_mm: float, subdivisions: int) -> Path:
    return Path("assets/objects") / f"sphere_r{radius_mm:g}_sub{subdivisions}.obj"


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius-mm", type=float, required=True)
    parser.add_argument("--subdivisions", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(arguments)

    mesh = make_sphere_mesh(args.radius_mm, args.subdivisions)
    output = args.output or _default_sphere_path(
        args.radius_mm,
        args.subdivisions,
    )
    save_obj(mesh, output)
    print(
        f"wrote {output} ({len(mesh.vertices_mm)} vertices, "
        f"{len(mesh.faces)} triangles, units=mm)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

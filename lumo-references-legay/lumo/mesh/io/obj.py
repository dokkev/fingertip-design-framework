"""Explicit OBJ asset import/export for neutral rigid meshes."""

from __future__ import annotations

from math import isfinite
from pathlib import Path

import numpy as np

from ..rigid.object import RigidObjectMesh


class RigidMeshAssetError(ValueError):
    """Raised when a rigid mesh asset cannot satisfy the neutral mesh contract."""


def _validate_scale(scale_mm_per_unit: float) -> float:
    scale = float(scale_mm_per_unit)
    if not isfinite(scale) or scale <= 0.0:
        raise RigidMeshAssetError(
            "scale_mm_per_unit must be finite and greater than zero"
        )
    return scale


def _obj_vertex_index(token: str, vertex_count: int) -> int:
    position = token.split("/", 1)[0]
    if not position:
        raise RigidMeshAssetError("OBJ face vertex references must include a position index")
    try:
        raw_index = int(position)
    except ValueError as exception:
        raise RigidMeshAssetError(
            f"invalid OBJ vertex index {position!r}"
        ) from exception
    if raw_index == 0:
        raise RigidMeshAssetError("OBJ vertex indices are one-based and cannot be zero")
    index = raw_index - 1 if raw_index > 0 else vertex_count + raw_index
    if index < 0 or index >= vertex_count:
        raise RigidMeshAssetError(
            f"OBJ vertex index {raw_index} is outside the loaded vertex list"
        )
    return index


def save_obj(mesh: RigidObjectMesh, path: str | Path) -> Path:
    """Write a neutral millimetre mesh to a triangulated OBJ file."""

    if not isinstance(mesh, RigidObjectMesh):
        raise TypeError("mesh must be RigidObjectMesh")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# LIT rigid object asset",
        "# units: millimetres",
        f"o {mesh.name}",
    ]
    lines.extend(
        f"v {float(x):.17g} {float(y):.17g} {float(z):.17g}"
        for x, y, z in mesh.vertices_mm
    )
    lines.extend(
        f"f {int(first) + 1} {int(second) + 1} {int(third) + 1}"
        for first, second, third in mesh.faces
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def load_obj(
    path: str | Path,
    *,
    scale_mm_per_unit: float,
    name: str | None = None,
) -> RigidObjectMesh:
    """Load a triangulated or polygonal OBJ into ``RigidObjectMesh``.

    OBJ position coordinates are multiplied by the explicit
    ``scale_mm_per_unit``. Texture and normal indices are ignored because the
    neutral runtime contract contains only closed triangle geometry.
    Polygonal faces are triangulated with a fan around their first vertex.
    ``RigidObjectMesh`` performs the final manifold, winding, and volume
    validation; no silent repair or recentering is performed here.
    """

    source = Path(path)
    if source.suffix.lower() != ".obj":
        raise RigidMeshAssetError(f"expected an OBJ file, got {source.name!r}")
    if not source.is_file():
        raise FileNotFoundError(source)
    scale = _validate_scale(scale_mm_per_unit)

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        record = fields[0]
        if record == "v":
            if len(fields) < 4:
                raise RigidMeshAssetError(
                    f"line {line_number}: OBJ vertex must contain three coordinates"
                )
            try:
                vertices.append(
                    (
                        float(fields[1]) * scale,
                        float(fields[2]) * scale,
                        float(fields[3]) * scale,
                    )
                )
            except ValueError as exception:
                raise RigidMeshAssetError(
                    f"line {line_number}: invalid OBJ vertex coordinates"
                ) from exception
            continue
        if record != "f":
            continue
        if len(fields) < 4:
            raise RigidMeshAssetError(
                f"line {line_number}: OBJ face must contain at least three vertices"
            )
        polygon = tuple(
            _obj_vertex_index(token, len(vertices)) for token in fields[1:]
        )
        faces.extend(
            (polygon[0], polygon[index], polygon[index + 1])
            for index in range(1, len(polygon) - 1)
        )

    if not vertices:
        raise RigidMeshAssetError(f"OBJ asset contains no vertices: {source}")
    if not faces:
        raise RigidMeshAssetError(f"OBJ asset contains no faces: {source}")

    resolved_name = name or source.stem
    try:
        return RigidObjectMesh(
            vertices_mm=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            name=resolved_name,
        )
    except (TypeError, ValueError) as exception:
        raise RigidMeshAssetError(
            f"OBJ asset {source} does not satisfy the closed rigid mesh contract: "
            f"{exception}"
        ) from exception


def load_obj_directory(
    directory: str | Path,
    *,
    scale_mm_per_unit: float,
) -> dict[str, RigidObjectMesh]:
    """Load all top-level OBJ assets in deterministic filename order."""

    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(root)
    assets: dict[str, RigidObjectMesh] = {}
    for path in sorted(root.glob("*.obj")):
        asset = load_obj(
            path,
            scale_mm_per_unit=scale_mm_per_unit,
            name=path.stem,
        )
        if asset.name in assets:
            raise RigidMeshAssetError(f"duplicate rigid asset name: {asset.name}")
        assets[asset.name] = asset
    return assets


__all__ = [
    "RigidMeshAssetError",
    "load_obj",
    "load_obj_directory",
    "save_obj",
]

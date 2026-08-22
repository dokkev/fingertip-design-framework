"""Sample feasible fingertips and validate both generated mesh components."""

from __future__ import annotations

from collections import Counter

import numpy as np

from lumo.fingertip import Fingertip, FingertipParameters
from lumo.mesh import make_fingertip_mesh
from lumo.optimization import (
    DesignParameterBounds,
    DesignSpace,
    LinearConstraint,
    ParameterBound,
)


def make_design_space() -> DesignSpace:
    """Return the fingertip design space used by this validation."""

    parameter_bounds = DesignParameterBounds(
        parameters=FingertipParameters(),
        geometry={
            "flat_pad_width_mm": ParameterBound(25.0, 35.0),
            "flat_pad_height_mm": ParameterBound(3.0, 8.0),
            "semiellipse_height_mm": ParameterBound(6.0, 20.0),
            "stem_width_mm": ParameterBound(7.0, 10.0),
            "void_width_mm": ParameterBound(0.0, 3.0),
            "void_height_mm": ParameterBound(0.0, 3.0),
        },
    )

    return DesignSpace(
        parameter_bounds=parameter_bounds,
        linear_constraints=(
            LinearConstraint(
                coefficients={
                    "geometry.flat_pad_height_mm": 1.0,
                    "geometry.semiellipse_height_mm": 1.0,
                },
                upper=30.0,
            ),
        ),
        minimum_silicone_thickness_mm=5.0,
    )


def sample_candidate(
    space: DesignSpace,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Sample one candidate uniformly from the raw parameter bounds."""

    candidate: dict[str, float] = {}

    for name, bound in space.parameter_bounds.geometry.items():
        candidate[f"geometry.{name}"] = float(
            rng.uniform(bound.lower, bound.upper)
        )

    for name, bound in space.parameter_bounds.led.items():
        candidate[f"led.{name}"] = float(
            rng.uniform(bound.lower, bound.upper)
        )

    return candidate


def signed_tet_volumes(mesh) -> np.ndarray:
    """Return signed volumes for all tetrahedra in a Newton mesh."""
    vertices = np.asarray(mesh.vertices)
    tetrahedra = np.asarray(mesh.tet_indices).reshape(-1, 4)

    x0 = vertices[tetrahedra[:, 0]]
    x1 = vertices[tetrahedra[:, 1]]
    x2 = vertices[tetrahedra[:, 2]]
    x3 = vertices[tetrahedra[:, 3]]

    return np.einsum(
        "ij,ij->i",
        x1 - x0,
        np.cross(x2 - x0, x3 - x0),
    ) / 6.0


def validate_tet_orientation(mesh) -> int:
    """Require finite, nonzero, and consistent tetrahedron orientation."""
    volumes = signed_tet_volumes(mesh)

    if np.any(~np.isfinite(volumes)) or np.any(volumes == 0.0):
        raise RuntimeError("mesh contains degenerate tetrahedra")

    signs = np.sign(volumes)
    if np.any(signs != signs[0]):
        raise RuntimeError("mesh contains inconsistent tetrahedron winding")

    return int(signs[0])


def validate_carrier_surface(mesh) -> int:
    """Require a closed, nondegenerate, outward-oriented carrier surface."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.indices).reshape(-1, 3)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError("carrier vertices must have shape (N, 3)")

    if not triangles.size:
        raise RuntimeError("carrier mesh contains no triangles")

    points = vertices[triangles]
    cross_products = np.cross(
        points[:, 1] - points[:, 0],
        points[:, 2] - points[:, 0],
    )
    twice_areas = np.linalg.norm(cross_products, axis=1)

    if np.any(~np.isfinite(twice_areas)) or np.any(twice_areas <= 0.0):
        raise RuntimeError("carrier mesh contains degenerate triangles")

    undirected_edges: Counter[tuple[int, int]] = Counter()
    directed_edges: Counter[tuple[int, int]] = Counter()

    for triangle in triangles:
        for start, end in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            undirected_edges[tuple(sorted((start, end)))] += 1
            directed_edges[(start, end)] += 1

    if any(count != 2 for count in undirected_edges.values()):
        raise RuntimeError("carrier mesh is not a closed surface")

    if any(
        directed_edges[(start, end)] != 1
        or directed_edges[(end, start)] != 1
        for start, end in undirected_edges
    ):
        raise RuntimeError("carrier mesh has inconsistent triangle winding")

    signed_volume = np.einsum(
        "ij,ij->i",
        points[:, 0],
        np.cross(points[:, 1], points[:, 2]),
    ).sum() / 6.0

    if not np.isfinite(signed_volume) or signed_volume <= 0.0:
        raise RuntimeError("carrier mesh does not have outward winding")

    return 1


def validate_bonded_vertices(mesh) -> int:
    """Require a nonempty, unique index set inside the silicone mesh."""
    indices = np.asarray(mesh.bonded_vertex_indices)

    if indices.ndim != 1 or indices.size == 0:
        raise RuntimeError("fingertip mesh contains no bonded vertices")

    if np.unique(indices).size != indices.size:
        raise RuntimeError("fingertip mesh contains duplicate bonded vertices")

    if np.any(indices < 0) or np.any(indices >= mesh.silicone.vertex_count):
        raise RuntimeError("fingertip mesh contains an invalid bonded index")

    return int(indices.size)


def main() -> None:
    rng = np.random.default_rng(0)
    space = make_design_space()

    target_meshes = 50
    max_attempts = 10_000

    extrusion_depth_mm = 11.0
    element_size_mm = 1.0

    vertex_counts: list[int] = []
    tet_counts: list[int] = []
    surface_triangle_counts: list[int] = []
    tet_orientation_signs: list[int] = []
    carrier_triangle_counts: list[int] = []
    carrier_orientation_signs: list[int] = []
    bonded_vertex_counts: list[int] = []

    feasible_count = 0
    mesh_failures = 0

    for attempt in range(1, max_attempts + 1):
        print(f"\rSampling fingertip mesh {attempt}/{target_meshes}...", end="")
        candidate = sample_candidate(space, rng)

        if not space.is_feasible(candidate):
            continue

        feasible_count += 1

        parameters = space.to_parameters(candidate)
        fingertip = Fingertip(parameters)

        try:
            mesh = make_fingertip_mesh(
                fingertip,
                extrusion_depth_mm=extrusion_depth_mm,
                element_size_mm=element_size_mm,
            )
        except Exception as exc:
            mesh_failures += 1

            print("\nMesh generation failed:")
            for name, value in candidate.items():
                print(f"  {name}: {value:.4f}")

            print(f"  error: {type(exc).__name__}: {exc}")
            continue

        try:
            tet_orientation_signs.append(
                validate_tet_orientation(mesh.silicone)
            )
            carrier_orientation_signs.append(
                validate_carrier_surface(mesh.carrier)
            )
            bonded_vertex_counts.append(validate_bonded_vertices(mesh))
        except Exception as exc:
            mesh_failures += 1

            print("\nMesh validation failed:")
            for name, value in candidate.items():
                print(f"  {name}: {value:.4f}")

            print(f"  error: {type(exc).__name__}: {exc}")
            continue

        vertex_counts.append(mesh.silicone.vertex_count)
        tet_counts.append(mesh.silicone.tet_count)
        surface_triangle_counts.append(
            len(mesh.silicone.surface_tri_indices) // 3
        )
        carrier_triangle_counts.append(
            len(np.asarray(mesh.carrier.indices)) // 3
        )

        if len(tet_counts) >= target_meshes:
            break

    print()
    print("Fingertip mesh sampling")
    print("-----------------------")
    print(f"Raw attempts:       {attempt}")
    print(f"Feasible samples:   {feasible_count}")
    print(f"Successful meshes:  {len(tet_counts)}")
    print(f"Mesh failures:      {mesh_failures}")

    if feasible_count > 0:
        print(
            f"Mesh success rate:  "
            f"{len(tet_counts) / feasible_count:.1%}"
        )

    if not tet_counts:
        raise RuntimeError(
            "no feasible fingertip produced a valid tetrahedral mesh"
        )

    if len(set(tet_orientation_signs)) != 1:
        raise RuntimeError(
            "successful meshes use inconsistent tetrahedron orientations"
        )

    if len(set(carrier_orientation_signs)) != 1:
        raise RuntimeError(
            "successful meshes use inconsistent carrier orientations"
        )

    orientation = "positive" if tet_orientation_signs[0] > 0 else "negative"
    print(f"Tet volume orientation: {orientation}")
    print("Carrier surface orientation: outward")

    print()
    print("Mesh size")
    print("---------")

    _print_statistics("vertices", vertex_counts)
    _print_statistics("tetrahedra", tet_counts)
    _print_statistics(
        "surface triangles",
        surface_triangle_counts,
    )
    _print_statistics(
        "carrier triangles",
        carrier_triangle_counts,
    )
    _print_statistics(
        "bonded vertices",
        bonded_vertex_counts,
    )


def _print_statistics(
    name: str,
    values: list[int],
) -> None:
    array = np.asarray(values)

    print(
        f"{name:18s}"
        f" min={array.min():6d}"
        f" mean={array.mean():8.1f}"
        f" max={array.max():6d}"
    )


if __name__ == "__main__":
    main()

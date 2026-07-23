"""Static dependency guards for the reusable package boundaries."""

from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _imports(package: str) -> set[str]:
    imported: set[str] = set()
    for path in (REPOSITORY_ROOT / package).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    return imported


def _assert_no_prefix(package: str, forbidden: tuple[str, ...]) -> None:
    violations = sorted(
        name
        for name in _imports(package)
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in forbidden
        )
    )
    assert not violations, f"{package} imports forbidden packages: {violations}"


def test_production_packages_do_not_import_validation_or_tests() -> None:
    for package in ("model", "mesh", "fem", "visualization"):
        _assert_no_prefix(package, ("validation", "tests"))


def test_model_is_geometry_only() -> None:
    _assert_no_prefix(
        "model",
        ("mesh", "fem", "visualization", "gmsh", "matplotlib", "KratosMultiphysics"),
    )


def test_mesh_is_solver_and_plotting_independent() -> None:
    _assert_no_prefix(
        "mesh",
        ("fem", "visualization", "matplotlib", "KratosMultiphysics"),
    )


def test_fem_has_no_plotting_dependency() -> None:
    _assert_no_prefix("fem", ("visualization", "matplotlib"))

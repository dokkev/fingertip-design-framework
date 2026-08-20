"""Static dependency guards for the current package boundaries."""

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
    for package in ("model", "mesh", "contact", "physics", "optics", "optimization"):
        _assert_no_prefix(package, ("validation", "tests"))


def test_model_is_geometry_only() -> None:
    _assert_no_prefix(
        "model",
        (
            "mesh",
            "fem",
            "physics",
            "optics",
            "visualization",
            "gmsh",
            "matplotlib",
            "KratosMultiphysics",
        ),
    )


def test_mesh_is_solver_and_plotting_independent() -> None:
    _assert_no_prefix(
        "mesh",
        ("fem", "physics", "optics", "validation", "visualization", "matplotlib", "KratosMultiphysics"),
    )


def test_physics_has_no_fem_or_optics_dependency() -> None:
    _assert_no_prefix("physics", ("fem", "optics", "validation", "tests"))


def test_contact_is_solver_independent() -> None:
    _assert_no_prefix("contact", ("physics", "validation", "tests"))


def test_optics_has_no_mechanics_or_validation_dependency() -> None:
    _assert_no_prefix("optics", ("physics", "validation", "tests"))


def test_lower_layers_do_not_import_gui() -> None:
    for package in ("model", "mesh", "contact", "physics", "optics", "optimization"):
        _assert_no_prefix(package, ("gui",))


def test_removed_legacy_packages_are_absent() -> None:
    for package in ("fem", "case", "visualization", "examples", "mechanics3d"):
        assert not any((REPOSITORY_ROOT / package).glob("*.py"))


def test_physics_has_one_flattened_newton_implementation() -> None:
    assert (REPOSITORY_ROOT / "physics" / "newton_vbd.py").is_file()
    assert not any((REPOSITORY_ROOT / "physics" / "backends").glob("*.py"))

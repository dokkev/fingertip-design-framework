"""Static dependency guards for the single ``lumo.*`` package namespace."""

from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LUMO_PACKAGE = REPOSITORY_ROOT / "lumo"


def _imports(package: str | None) -> set[str]:
    imported: set[str] = set()
    paths = (
        LUMO_PACKAGE.glob("*.py")
        if package is None
        else (LUMO_PACKAGE / package).rglob("*.py")
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.level == 0
            ):
                imported.add(node.module)
    return imported


def _assert_no_prefix(package: str | None, forbidden: tuple[str, ...]) -> None:
    violations = sorted(
        name
        for name in _imports(package)
        if any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in forbidden
        )
    )
    owner = "lumo" if package is None else f"lumo.{package}"
    assert not violations, f"{owner} imports forbidden packages: {violations}"


def test_production_packages_do_not_import_validation_or_tests() -> None:
    for package in (
        None,
        "finger",
        "mesh",
        "contact",
        "physics",
        "ray_tracing",
        "optimization",
        "util",
    ):
        _assert_no_prefix(package, ("validation", "tests"))


def test_model_is_geometry_only() -> None:
    _assert_no_prefix(
        "finger",
        (
            "lumo.mesh",
            "fem",
            "lumo.physics",
            "lumo.ray_tracing",
            "visualization",
            "gmsh",
            "matplotlib",
            "KratosMultiphysics",
        ),
    )


def test_mesh_is_solver_and_plotting_independent() -> None:
    _assert_no_prefix(
        "mesh",
        (
            "fem",
            "lumo.physics",
            "lumo.ray_tracing",
            "validation",
            "visualization",
            "matplotlib",
            "KratosMultiphysics",
        ),
    )


def test_physics_has_no_fem_or_optics_dependency() -> None:
    _assert_no_prefix(
        "physics",
        ("fem", "lumo.ray_tracing", "validation", "tests"),
    )


def test_contact_is_solver_independent() -> None:
    _assert_no_prefix("contact", ("lumo.physics", "validation", "tests"))


def test_optics_has_no_mechanics_or_validation_dependency() -> None:
    _assert_no_prefix(
        "ray_tracing",
        ("lumo.physics", "validation", "tests"),
    )


def test_lower_layers_do_not_import_gui() -> None:
    for package in (
        None,
        "finger",
        "mesh",
        "contact",
        "physics",
        "ray_tracing",
        "optimization",
        "util",
    ):
        _assert_no_prefix(package, ("gui",))


def test_lumo_does_not_import_optimization() -> None:
    _assert_no_prefix(None, ("lumo.optimization",))


def test_removed_legacy_packages_are_absent() -> None:
    for package in (
        "fem",
        "case",
        "examples",
        "mechanics3d",
        "model",
        "optics",
    ):
        assert not (REPOSITORY_ROOT / package).exists()
    assert not (LUMO_PACKAGE / "mesh" / "fingertip").exists()


def test_framework_code_has_one_installable_namespace() -> None:
    assert (LUMO_PACKAGE / "__init__.py").is_file()
    for package in (
        "finger",
        "mesh",
        "contact",
        "physics",
        "ray_tracing",
        "optimization",
        "util",
    ):
        assert not (REPOSITORY_ROOT / package).exists()


def test_noncanonical_fingertip_adapters_have_role_specific_names() -> None:
    assert (LUMO_PACKAGE / "physics" / "trajectory" / "fingertip_adapter.py").is_file()
    assert (
        LUMO_PACKAGE / "ray_tracing" / "optical_mechanics" / "state_adapter.py"
    ).is_file()
    assert not (LUMO_PACKAGE / "physics" / "trajectory" / "fingertip.py").exists()
    assert not (
        LUMO_PACKAGE / "ray_tracing" / "optical_mechanics" / "fingertip.py"
    ).exists()


def test_visualization_config_is_repository_only() -> None:
    plot_config = REPOSITORY_ROOT / "visualization" / "config" / "lumo_plot.yaml"
    assert plot_config.is_file()
    assert not any((REPOSITORY_ROOT / "visualization").rglob("*.py"))


def test_physics_has_one_flattened_newton_implementation() -> None:
    assert (LUMO_PACKAGE / "physics" / "newton" / "vbd.py").is_file()
    assert not any((LUMO_PACKAGE / "physics" / "backends").glob("*.py"))

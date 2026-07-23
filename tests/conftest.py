"""Path-based pytest markers keep dependency boundaries explicit."""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = Path(str(item.path))
        parts = set(path.parts)
        if "smoke" in parts:
            item.add_marker(pytest.mark.smoke)
        if "gmsh" in parts:
            item.add_marker(pytest.mark.gmsh)
        if "kratos" in parts:
            item.add_marker(pytest.mark.kratos)

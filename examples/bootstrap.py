"""Make repository packages importable from directly executed examples."""

from __future__ import annotations

from pathlib import Path
import sys


def ensure_repository_root(start: Path | None = None) -> Path:
    """Add the repository root above ``start`` to ``sys.path`` and return it."""
    search_start = (start or Path(__file__).resolve().parent).resolve()
    for candidate in (search_start, *search_start.parents):
        if (
            (candidate / "model" / "__init__.py").is_file()
            and (candidate / "visualization" / "__init__.py").is_file()
        ):
            repository_root = str(candidate)
            if repository_root not in sys.path:
                sys.path.insert(0, repository_root)
            return candidate

    raise RuntimeError(
        f"Could not locate the repository root above {search_start}. "
        "Expected model and visualization packages."
    )

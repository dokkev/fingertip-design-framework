"""Repository and artifact provenance helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_revision(repository_root: str | Path) -> str | None:
    """Return HEAD without making repository state a runtime requirement."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repository_root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def file_provenance(path: str | Path) -> dict[str, Any]:
    """Return stable path, size, and digest metadata."""
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }

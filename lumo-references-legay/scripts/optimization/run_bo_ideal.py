"""Canonical human-facing entry point for stable production LUMO BO.

The readable workflow is intentionally small:

1. load and strictly validate ``config/lumo_execution.yaml`` once;
2. resolve explicit success targets and independent hard caps;
3. verify source/cache policy before expensive preflight work;
4. execute the checkpointed production campaign engine;
5. return nonzero for infrastructure or controlled scientific failure.

Scientific morphology, protocol, objective, persistence, and resume ownership
remain in :mod:`scripts.optimization.run_bo`; this file is the supported CLI,
not a second implementation or a future-API sketch.
"""

from __future__ import annotations

from typing import Sequence

from scripts.optimization.run_bo import main as _run_bo_main


def main(argv: Sequence[str] | None = None) -> int:
    """Run the production CLI using the single campaign implementation."""

    return _run_bo_main(None if argv is None else list(argv))


if __name__ == "__main__":
    raise SystemExit(main())

"""Backward-compatible entry point for the shared OptiX smoke contract."""

from __future__ import annotations

import sys

# Keep the shared runtime boundary visible to callers and source-level checks.
from optics.optix.runtime import OptixRuntime  # noqa: F401
from optics.optix.smoke import (
    ProductionOptixSmokeError,
    ProductionOptixSmokeResult,
    run_production_optix_smoke,
)


SmokeResult = ProductionOptixSmokeResult


def run_smoke() -> tuple[SmokeResult, dict[str, object]]:
    """Run the real shared smoke and return its result and runtime metadata."""
    result = run_production_optix_smoke()
    return result, dict(result.metadata)


def main() -> int:
    try:
        result, metadata = run_smoke()
    except ProductionOptixSmokeError as exc:
        print(f"FAIL OptiX smoke [{exc.stage}]: {exc}", file=sys.stderr)
        return 1
    print(
        "PASS OptiX smoke: "
        f"OptiX {metadata['optix_version']}; "
        f"CUDA device {metadata['cuda_device']} "
        f"(compute capability {metadata['compute_capability']}); "
        f"OptiX include {metadata['optix_include']}; "
        f"CUDA include {metadata['cuda_include']}; "
        f"hit={result.hit}; miss={result.miss}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

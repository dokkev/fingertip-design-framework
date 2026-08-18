"""Run the production OptiX preflight with one real deterministic launch."""

from __future__ import annotations

import sys

from optics.optix.smoke import (
    ProductionOptixSmokeError,
    run_production_optix_smoke,
)


def _version(value: object) -> str:
    if isinstance(value, (tuple, list)):
        return ".".join(str(item) for item in value)
    return str(value)


def main() -> int:
    try:
        result = run_production_optix_smoke()
    except ProductionOptixSmokeError as exc:
        print(f"FAIL: {exc.stage}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(
            "FAIL: optix_runtime_initialization: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    metadata = result.metadata
    terminal = ",".join(
        f"{name}={count}" for name, count in result.terminal_event_counts.items()
    )
    results = ",".join(
        f"{name}={count}" for name, count in result.result_counts.items()
    )
    print(
        "PASS: production_optix_smoke "
        f"GPU/device={metadata['cuda_device']} "
        f"OptiX={metadata['optix_version']} "
        f"CUDA_runtime={metadata['cuda_runtime_version']} "
        f"NVRTC={_version(metadata['nvrtc_version'])} "
        f"OptiX_include={metadata['optix_include']} "
        f"CUDA_include={metadata['cuda_include']} "
        f"setup={result.setup_time_seconds:.3f}s "
        f"trace={result.trace_time_seconds:.3f}s "
        f"rays={result.ray_count} "
        f"terminal={terminal} results={results}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

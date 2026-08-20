"""Dependency-light diagnostics for the optional OptiX environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

from optics.optix._paths import _cuda_candidates, _optix_candidates


def _header_status(
    candidates: tuple[Path, ...],
    required_headers: tuple[str, ...],
) -> dict[str, Any]:
    records = []
    resolved: Path | None = None
    for candidate in candidates:
        missing = [
            header
            for header in required_headers
            if not (candidate / header).is_file()
        ]
        records.append(
            {
                "path": str(candidate),
                "missing_required_headers": missing,
            }
        )
        if resolved is None and candidate.is_dir() and not missing:
            resolved = candidate.resolve()
    return {
        "resolved": resolved is not None,
        "directory": None if resolved is None else str(resolved),
        "candidates": records,
    }


def _diagnose_include_paths() -> dict[str, Any]:
    return {
        "optix": _header_status(
            _optix_candidates(os.environ),
            ("optix.h", "optix_device.h"),
        ),
        "cuda": _header_status(
            _cuda_candidates(os.environ),
            ("cuda.h", "cuda_runtime.h"),
        ),
    }


def _module_status(name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:
        return {
            "importable": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if spec is None:
        return {"importable": False, "error": "module not found"}
    distribution = {
        "optix": "pyoptix",
        "cupy": "cupy",
        "cuda.bindings.nvrtc": "cuda-python",
    }.get(name)
    try:
        version = (
            importlib.metadata.version(distribution)
            if distribution is not None
            else None
        )
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "importable": True,
        "location": str(Path(spec.origin).resolve())
        if spec.origin and spec.origin not in ("built-in", "frozen")
        else spec.origin,
        "version": version,
    }


def _gpu_runtime_status() -> dict[str, Any]:
    try:
        import cupy as cp

        device = cp.cuda.Device()
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        return {
            "available": True,
            "device": name,
            "compute_capability": [properties["major"], properties["minor"]],
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_diagnostics() -> dict[str, Any]:
    """Collect import, header, and optional GPU runtime status."""
    return {
        "python_executable": sys.executable,
        "pyoptix": _module_status("optix"),
        "cupy": _module_status("cupy"),
        "cuda_python_nvrtc": _module_status("cuda.bindings.nvrtc"),
        "headers": _diagnose_include_paths(),
        "gpu_runtime": _gpu_runtime_status(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON diagnostics")
    arguments = parser.parse_args(argv)
    diagnostics = collect_diagnostics()
    if arguments.json:
        print(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"Python executable: {diagnostics['python_executable']}")
        for name in ("pyoptix", "cupy", "cuda_python_nvrtc"):
            print(f"{name}: {diagnostics[name]}")
        for name, values in diagnostics["headers"].items():
            print(f"{name} header resolution: {values}")
        print(f"GPU runtime: {diagnostics['gpu_runtime']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

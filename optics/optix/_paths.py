"""Private CUDA and OptiX include-directory resolution for the runtime."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
from typing import Mapping


@dataclass(frozen=True)
class _IncludePaths:
    optix: Path
    cuda: Path
    optix_source: str
    cuda_source: str


def _unique_paths(candidates: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    result: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for candidate, source in candidates:
        resolved = candidate.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            result.append((resolved, source))
    return result


def _optix_candidates(
    environment: Mapping[str, str],
    explicit_include_dir: str | os.PathLike[str] | None = None,
) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    if explicit_include_dir is not None:
        candidates.append((Path(explicit_include_dir), "explicit argument"))
    direct = environment.get("OPTIX_INCLUDE_DIR")
    if direct:
        candidates.append((Path(direct), "OPTIX_INCLUDE_DIR"))
    for variable in ("OptiX_INSTALL_DIR", "OPTIX_ROOT"):
        value = environment.get(variable)
        if value:
            root = Path(value)
            candidates.extend(
                (
                    (root, f"{variable} (root)"),
                    (root / "include", f"{variable} (include)"),
                )
            )
    candidates.extend(
        (
            (Path("/usr/local/OptiX/include"), "conventional location"),
            (Path("/usr/local/optix/include"), "conventional location"),
            (Path("/opt/optix/include"), "conventional location"),
        )
    )
    return _unique_paths(candidates)


def _cuda_candidates(
    environment: Mapping[str, str],
    explicit_include_dir: str | os.PathLike[str] | None = None,
) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    if explicit_include_dir is not None:
        candidates.append((Path(explicit_include_dir), "explicit argument"))
    direct = environment.get("CUDA_INCLUDE_DIR")
    if direct:
        candidates.append((Path(direct), "CUDA_INCLUDE_DIR"))

    roots: list[tuple[Path, str]] = []
    for variable in ("CONDA_PREFIX", "CUDA_HOME", "CUDA_PATH"):
        value = environment.get(variable)
        if value:
            roots.append((Path(value), variable))
    if sys.prefix:
        roots.append((Path(sys.prefix), "Python sys.prefix"))

    for root, source in roots:
        candidates.extend(
            (
                (
                    root / "targets" / "x86_64-linux" / "include",
                    f"{source} (x86_64 targets)",
                ),
                (
                    root / "targets" / "aarch64-linux" / "include",
                    f"{source} (aarch64 targets)",
                ),
                (root / "include", f"{source} (include)"),
            )
        )

    nvcc = shutil.which("nvcc")
    if nvcc:
        nvcc_root = Path(nvcc).resolve().parent.parent
        candidates.extend(
            (
                (
                    nvcc_root / "targets" / "x86_64-linux" / "include",
                    "nvcc (x86_64 targets)",
                ),
                (nvcc_root / "include", "nvcc (include)"),
            )
        )
    candidates.append((Path("/usr/local/cuda/include"), "conventional location"))
    return _unique_paths(candidates)


def _resolve(
    candidates: list[tuple[Path, str]],
    required_headers: tuple[str, ...],
) -> tuple[Path | None, str | None, list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    for candidate, source in candidates:
        missing = tuple(
            header
            for header in required_headers
            if not (candidate / header).is_file()
        )
        records.append(
            {
                "path": str(candidate),
                "source": source,
                "missing_required_headers": list(missing),
            }
        )
        if candidate.is_dir() and not missing:
            return candidate.resolve(), source, records
    return None, None, records


def _format_failure(
    label: str,
    required_headers: tuple[str, ...],
    candidates: list[dict[str, object]],
    variables: tuple[str, ...],
) -> str:
    searched = "; ".join(
        f"{record['path']} [{record['source']}; missing: "
        f"{', '.join(record['missing_required_headers']) or 'none'}]"
        for record in candidates
    ) or "<none>"
    return (
        f"Could not find a valid {label} include directory. "
        f"Required headers: {', '.join(required_headers)}. "
        f"Candidates: {searched}. Set {', '.join(variables)} to the exact "
        "directory containing the required headers."
    )


def _diagnose_include_paths(
    environment: Mapping[str, str] | None = None,
    *,
    optix_include_dir: str | os.PathLike[str] | None = None,
    cuda_include_dir: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Return header diagnostics for the tooling-layer doctor."""
    environment = os.environ if environment is None else environment
    optix, optix_source, optix_candidates = _resolve(
        _optix_candidates(environment, optix_include_dir),
        ("optix.h", "optix_device.h"),
    )
    cuda, cuda_source, cuda_candidates = _resolve(
        _cuda_candidates(environment, cuda_include_dir),
        ("cuda.h", "cuda_runtime.h"),
    )
    return {
        "optix": {
            "resolved": optix is not None,
            "directory": None if optix is None else str(optix),
            "source": optix_source,
            "candidates": optix_candidates,
        },
        "cuda": {
            "resolved": cuda is not None,
            "directory": None if cuda is None else str(cuda),
            "source": cuda_source,
            "candidates": cuda_candidates,
        },
    }


def _discover_include_paths(
    environment: Mapping[str, str] | None = None,
    *,
    optix_include_dir: str | os.PathLike[str] | None = None,
    cuda_include_dir: str | os.PathLike[str] | None = None,
) -> _IncludePaths:
    """Resolve the two include directories required by NVRTC."""
    diagnostics = _diagnose_include_paths(
        environment,
        optix_include_dir=optix_include_dir,
        cuda_include_dir=cuda_include_dir,
    )
    optix = diagnostics["optix"]
    cuda = diagnostics["cuda"]
    if not optix["resolved"]:
        raise RuntimeError(
            _format_failure(
                "OptiX",
                ("optix.h", "optix_device.h"),
                optix["candidates"],
                ("OPTIX_INCLUDE_DIR", "OptiX_INSTALL_DIR", "OPTIX_ROOT"),
            )
        )
    if not cuda["resolved"]:
        raise RuntimeError(
            _format_failure(
                "CUDA",
                ("cuda.h", "cuda_runtime.h"),
                cuda["candidates"],
                ("CUDA_INCLUDE_DIR", "CONDA_PREFIX", "CUDA_HOME", "CUDA_PATH"),
            )
        )
    return _IncludePaths(
        optix=Path(str(optix["directory"])),
        cuda=Path(str(cuda["directory"])),
        optix_source=str(optix["source"]),
        cuda_source=str(cuda["source"]),
    )


__all__ = ["_IncludePaths", "_diagnose_include_paths", "_discover_include_paths"]

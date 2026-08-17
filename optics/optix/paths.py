"""Environment-aware discovery of optional CUDA and OptiX headers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
from typing import Mapping


@dataclass(frozen=True)
class IncludeCandidate:
    """One non-recursive include candidate and its rejection reason."""

    path: Path
    source: str
    missing_headers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "source": self.source,
            "missing_required_headers": list(self.missing_headers),
        }


@dataclass(frozen=True)
class IncludeResolution:
    """Resolution result used by the doctor and failure diagnostics."""

    directory: Path | None
    source: str | None
    candidates: tuple[IncludeCandidate, ...]

    @property
    def resolved(self) -> bool:
        return self.directory is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "resolved": self.resolved,
            "directory": None if self.directory is None else str(self.directory),
            "source": self.source,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class OptixCudaPaths:
    """Validated include directories required by the optional OptiX path."""

    optix_include: Path
    cuda_include: Path
    optix_resolution_source: str = ""
    cuda_resolution_source: str = ""


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
    for path in (
        "/usr/local/OptiX/include",
        "/usr/local/optix/include",
        "/opt/optix/include",
    ):
        candidates.append((Path(path), "conventional location"))
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


def _resolve_include(
    candidates: list[tuple[Path, str]],
    required_headers: tuple[str, ...],
) -> IncludeResolution:
    records: list[IncludeCandidate] = []
    for candidate, source in candidates:
        missing = tuple(
            header
            for header in required_headers
            if not (candidate / header).is_file()
        )
        records.append(IncludeCandidate(candidate, source, missing))
        if candidate.is_dir() and not missing:
            return IncludeResolution(candidate.resolve(), source, tuple(records))
    return IncludeResolution(None, None, tuple(records))


def _format_resolution_failure(
    label: str,
    required_headers: tuple[str, ...],
    resolution: IncludeResolution,
    variables: tuple[str, ...],
) -> str:
    searched = "; ".join(
        f"{record.path} [{record.source}; missing: "
        f"{', '.join(record.missing_headers) or 'none'}]"
        for record in resolution.candidates
    ) or "<none>"
    return (
        f"Could not find a valid {label} include directory. "
        f"Required headers: {', '.join(required_headers)}. "
        f"Candidates: {searched}. Set {', '.join(variables)} to the exact "
        "directory containing the required headers."
    )


def diagnose_paths(
    environment: Mapping[str, str] | None = None,
    *,
    optix_include_dir: str | os.PathLike[str] | None = None,
    cuda_include_dir: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Return dependency-free header resolution diagnostics without raising."""
    environment = os.environ if environment is None else environment
    optix = _resolve_include(
        _optix_candidates(environment, optix_include_dir),
        ("optix.h", "optix_device.h"),
    )
    cuda = _resolve_include(
        _cuda_candidates(environment, cuda_include_dir),
        ("cuda.h", "cuda_runtime.h"),
    )
    return {"optix": optix.to_dict(), "cuda": cuda.to_dict()}


def discover_paths(
    environment: Mapping[str, str] | None = None,
    *,
    optix_include_dir: str | os.PathLike[str] | None = None,
    cuda_include_dir: str | os.PathLike[str] | None = None,
) -> OptixCudaPaths:
    """Resolve required headers using explicit, environment, then defaults."""
    environment = os.environ if environment is None else environment
    optix_resolution = _resolve_include(
        _optix_candidates(environment, optix_include_dir),
        ("optix.h", "optix_device.h"),
    )
    if not optix_resolution.resolved:
        raise RuntimeError(
            _format_resolution_failure(
                "OptiX",
                ("optix.h", "optix_device.h"),
                optix_resolution,
                ("OPTIX_INCLUDE_DIR", "OptiX_INSTALL_DIR", "OPTIX_ROOT"),
            )
        )
    cuda_resolution = _resolve_include(
        _cuda_candidates(environment, cuda_include_dir),
        ("cuda.h", "cuda_runtime.h"),
    )
    if not cuda_resolution.resolved:
        raise RuntimeError(
            _format_resolution_failure(
                "CUDA",
                ("cuda.h", "cuda_runtime.h"),
                cuda_resolution,
                ("CUDA_INCLUDE_DIR", "CONDA_PREFIX", "CUDA_HOME", "CUDA_PATH"),
            )
        )
    return OptixCudaPaths(
        optix_include=optix_resolution.directory,
        cuda_include=cuda_resolution.directory,
        optix_resolution_source=str(optix_resolution.source),
        cuda_resolution_source=str(cuda_resolution.source),
    )


__all__ = [
    "IncludeCandidate",
    "IncludeResolution",
    "OptixCudaPaths",
    "diagnose_paths",
    "discover_paths",
]

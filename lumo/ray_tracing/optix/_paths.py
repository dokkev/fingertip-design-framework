"""Private CUDA and OptiX include-directory resolution for production NVRTC."""

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


def _unique_paths(candidates: list[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded not in seen:
            seen.add(expanded)
            result.append(expanded)
    return tuple(result)


def _optix_candidates(
    environment: Mapping[str, str],
    explicit_include_dir: str | os.PathLike[str] | None = None,
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if explicit_include_dir is not None:
        candidates.append(Path(explicit_include_dir))
    direct = environment.get("OPTIX_INCLUDE_DIR")
    if direct:
        candidates.append(Path(direct))
    for variable in ("OptiX_INSTALL_DIR", "OPTIX_ROOT"):
        value = environment.get(variable)
        if value:
            root = Path(value)
            candidates.extend((root, root / "include"))
    candidates.extend(
        (
            Path("/usr/local/OptiX/include"),
            Path("/usr/local/optix/include"),
            Path("/opt/optix/include"),
        )
    )
    return _unique_paths(candidates)


def _cuda_candidates(
    environment: Mapping[str, str],
    explicit_include_dir: str | os.PathLike[str] | None = None,
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if explicit_include_dir is not None:
        candidates.append(Path(explicit_include_dir))
    direct = environment.get("CUDA_INCLUDE_DIR")
    if direct:
        candidates.append(Path(direct))

    roots = [
        Path(value)
        for variable in ("CONDA_PREFIX", "CUDA_HOME", "CUDA_PATH")
        if (value := environment.get(variable))
    ]
    if sys.prefix:
        roots.append(Path(sys.prefix))
    for root in roots:
        candidates.extend(
            (
                root / "targets" / "x86_64-linux" / "include",
                root / "targets" / "aarch64-linux" / "include",
                root / "include",
            )
        )

    nvcc = shutil.which("nvcc")
    if nvcc:
        nvcc_root = Path(nvcc).resolve().parent.parent
        candidates.extend(
            (
                nvcc_root / "targets" / "x86_64-linux" / "include",
                nvcc_root / "include",
            )
        )
    candidates.append(Path("/usr/local/cuda/include"))
    return _unique_paths(candidates)


def _resolve(
    candidates: tuple[Path, ...],
    required_headers: tuple[str, ...],
) -> Path | None:
    for candidate in candidates:
        if candidate.is_dir() and all(
            (candidate / header).is_file() for header in required_headers
        ):
            return candidate.resolve()
    return None


def _missing_include_message(
    label: str,
    candidates: tuple[Path, ...],
    required_headers: tuple[str, ...],
    variables: tuple[str, ...],
) -> str:
    return (
        f"Could not find a valid {label} include directory. "
        f"Required headers: {', '.join(required_headers)}. "
        f"Searched: {', '.join(str(path) for path in candidates) or '<none>'}. "
        f"Set {', '.join(variables)} to the directory containing those headers."
    )


def _discover_include_paths(
    environment: Mapping[str, str] | None = None,
    *,
    optix_include_dir: str | os.PathLike[str] | None = None,
    cuda_include_dir: str | os.PathLike[str] | None = None,
) -> _IncludePaths:
    """Resolve only the two include directories required by NVRTC."""
    selected_environment = os.environ if environment is None else environment
    optix_candidates = _optix_candidates(
        selected_environment,
        optix_include_dir,
    )
    cuda_candidates = _cuda_candidates(
        selected_environment,
        cuda_include_dir,
    )
    optix = _resolve(optix_candidates, ("optix.h", "optix_device.h"))
    if optix is None:
        raise RuntimeError(
            _missing_include_message(
                "OptiX",
                optix_candidates,
                ("optix.h", "optix_device.h"),
                ("OPTIX_INCLUDE_DIR", "OptiX_INSTALL_DIR", "OPTIX_ROOT"),
            )
        )
    cuda = _resolve(cuda_candidates, ("cuda.h", "cuda_runtime.h"))
    if cuda is None:
        raise RuntimeError(
            _missing_include_message(
                "CUDA",
                cuda_candidates,
                ("cuda.h", "cuda_runtime.h"),
                ("CUDA_INCLUDE_DIR", "CONDA_PREFIX", "CUDA_HOME", "CUDA_PATH"),
            )
        )
    return _IncludePaths(optix=optix, cuda=cuda)


__all__ = ["_IncludePaths", "_discover_include_paths"]

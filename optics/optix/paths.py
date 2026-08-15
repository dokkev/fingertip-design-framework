"""Environment-aware discovery of external CUDA and OptiX headers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
from typing import Mapping


@dataclass(frozen=True)
class OptixCudaPaths:
    """Validated include directories required by the OptiX smoke path."""

    optix_include: Path
    cuda_include: Path


def _unique_paths(candidates: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _optix_candidates(environment: Mapping[str, str]) -> list[Path]:
    candidates: list[Path] = []
    for variable in ("OptiX_INSTALL_DIR", "OPTIX_ROOT"):
        value = environment.get(variable)
        if value:
            root = Path(value)
            candidates.extend((root, root / "include"))
    return _unique_paths(candidates)


def _cuda_candidates(environment: Mapping[str, str]) -> list[Path]:
    candidates: list[Path] = []
    roots: list[Path] = []
    for variable in ("CONDA_PREFIX", "CUDA_HOME", "CUDA_PATH"):
        value = environment.get(variable)
        if value:
            roots.append(Path(value))
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


def _validated_include(
    *,
    candidates: list[Path],
    required_headers: tuple[str, ...],
    label: str,
    variables: tuple[str, ...],
) -> Path:
    for candidate in candidates:
        if candidate.is_dir() and all(
            (candidate / header).is_file() for header in required_headers
        ):
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates) or "<none>"
    variable_text = " or ".join(variables)
    headers = ", ".join(required_headers)
    raise RuntimeError(
        f"Could not find a valid {label} include directory. Searched: {searched}. "
        f"Required headers: {headers}. Set {variable_text} to the installation "
        "root and retry."
    )


def discover_paths(
    environment: Mapping[str, str] | None = None,
) -> OptixCudaPaths:
    """Discover and validate external OptiX and CUDA include directories."""
    environment = os.environ if environment is None else environment
    optix_include = _validated_include(
        candidates=_optix_candidates(environment),
        required_headers=("optix.h", "optix_device.h"),
        label="OptiX",
        variables=("OptiX_INSTALL_DIR", "OPTIX_ROOT"),
    )
    cuda_include = _validated_include(
        candidates=_cuda_candidates(environment),
        required_headers=("cuda.h", "cuda_runtime.h"),
        label="CUDA",
        variables=("CONDA_PREFIX", "CUDA_HOME", "CUDA_PATH"),
    )
    return OptixCudaPaths(
        optix_include=optix_include,
        cuda_include=cuda_include,
    )

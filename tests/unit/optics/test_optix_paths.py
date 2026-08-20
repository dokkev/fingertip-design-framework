from __future__ import annotations

import json
import inspect
from pathlib import Path

from scripts.tools.optix_doctor import main as doctor_main
from optics.optix._paths import _diagnose_include_paths, _discover_include_paths
from validation.optics import optix_smoke


def _headers(directory: Path, names: tuple[str, ...]) -> None:
    directory.mkdir(parents=True)
    for name in names:
        (directory / name).write_text("// synthetic header\n")


def test_direct_include_overrides_are_resolved_first(tmp_path, monkeypatch) -> None:
    optix = tmp_path / "optix-include"
    cuda = tmp_path / "cuda-include"
    _headers(optix, ("optix.h", "optix_device.h"))
    _headers(cuda, ("cuda.h", "cuda_runtime.h"))
    monkeypatch.delenv("OPTIX_INCLUDE_DIR", raising=False)
    monkeypatch.delenv("CUDA_INCLUDE_DIR", raising=False)
    paths = _discover_include_paths(
        {},
        optix_include_dir=optix,
        cuda_include_dir=cuda,
    )
    assert paths.optix == optix.resolve()
    assert paths.cuda == cuda.resolve()
    assert paths.optix_source == "explicit argument"
    assert paths.cuda_source == "explicit argument"


def test_environment_direct_include_overrides_are_supported(tmp_path) -> None:
    optix = tmp_path / "optix-include"
    cuda = tmp_path / "cuda-include"
    _headers(optix, ("optix.h", "optix_device.h"))
    _headers(cuda, ("cuda.h", "cuda_runtime.h"))
    paths = _discover_include_paths(
        {
            "OPTIX_INCLUDE_DIR": str(optix),
            "CUDA_INCLUDE_DIR": str(cuda),
        }
    )
    assert paths.optix == optix.resolve()
    assert paths.cuda == cuda.resolve()
    assert paths.optix_source == "OPTIX_INCLUDE_DIR"
    assert paths.cuda_source == "CUDA_INCLUDE_DIR"


def test_optix_install_root_and_optix_root_include_are_supported(tmp_path) -> None:
    install_root = tmp_path / "install"
    optix_install_include = install_root / "include"
    _headers(optix_install_include, ("optix.h", "optix_device.h"))
    cuda = tmp_path / "cuda"
    _headers(cuda, ("cuda.h", "cuda_runtime.h"))
    paths = _discover_include_paths(
        {
            "OptiX_INSTALL_DIR": str(install_root),
            "CUDA_INCLUDE_DIR": str(cuda),
        }
    )
    assert paths.optix == optix_install_include.resolve()

    root = tmp_path / "root"
    root_include = root / "include"
    _headers(root_include, ("optix.h", "optix_device.h"))
    root_paths = _discover_include_paths(
        {
            "OPTIX_ROOT": str(root),
            "CUDA_INCLUDE_DIR": str(cuda),
        }
    )
    assert root_paths.optix == root_include.resolve()


def test_missing_header_diagnostic_lists_candidate_source_and_headers(tmp_path) -> None:
    optix = tmp_path / "incomplete"
    optix.mkdir()
    diagnostics = _diagnose_include_paths(
        {
            "OPTIX_INCLUDE_DIR": str(optix),
            "CUDA_INCLUDE_DIR": str(tmp_path / "missing-cuda"),
        }
    )
    optix_candidates = diagnostics["optix"]["candidates"]
    assert optix_candidates[0]["path"] == str(optix)
    assert optix_candidates[0]["source"] == "OPTIX_INCLUDE_DIR"
    assert optix_candidates[0]["missing_required_headers"] == [
        "optix.h",
        "optix_device.h",
    ]
    try:
        _discover_include_paths(
            {
                "OPTIX_INCLUDE_DIR": str(optix),
                "CUDA_INCLUDE_DIR": str(tmp_path / "missing-cuda"),
            }
        )
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("missing headers should fail closed")
    assert str(optix) in message
    assert "OPTIX_INCLUDE_DIR" in message
    assert "optix_device.h" in message


def test_doctor_json_is_dependency_light(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPTIX_INCLUDE_DIR", "/definitely/missing/optix")
    monkeypatch.setenv("CUDA_INCLUDE_DIR", "/definitely/missing/cuda")
    assert doctor_main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["python_executable"]
    assert "headers" in payload
    assert payload["headers"]["optix"]["resolved"] is False
    assert any(
        candidate["path"] == "/definitely/missing/cuda"
        for candidate in payload["headers"]["cuda"]["candidates"]
    )


def test_validation_smoke_uses_shared_runtime_boundary() -> None:
    source = inspect.getsource(optix_smoke)
    assert "OptixRuntime" in source
    assert "__raygen__raygen_program" in source

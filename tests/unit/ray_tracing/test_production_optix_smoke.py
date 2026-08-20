from __future__ import annotations

import inspect
from types import ModuleType, SimpleNamespace
import sys

import numpy as np

from scripts.tools import optix_smoke as smoke


class _FakeArray:
    def __init__(self, values: np.ndarray, owner: "_FakeCuPy") -> None:
        self.values = values
        self.data = SimpleNamespace(ptr=id(self))
        self.nbytes = values.nbytes
        owner.arrays[self.data.ptr] = self

    def fill(self, value: int) -> None:
        self.values.fill(value)


class _FakeCuPy:
    uint32 = np.uint32

    def __init__(self) -> None:
        self.arrays: dict[int, _FakeArray] = {}

        class _Device:
            id = 0

            def use(self) -> None:
                return None

        class _Runtime:
            @staticmethod
            def getDeviceProperties(device_id: int) -> dict[str, object]:
                return {"name": b"synthetic GPU", "major": 8, "minor": 9}

            @staticmethod
            def runtimeGetVersion() -> int:
                return 12040

        self.cuda = SimpleNamespace(
            Device=_Device,
            runtime=_Runtime,
        )

    def zeros(self, size: int, *, dtype: object) -> _FakeArray:
        return _FakeArray(np.zeros(size, dtype=dtype), self)

    def asnumpy(self, array: _FakeArray) -> np.ndarray:
        return array.values.copy()


class _FakeRuntime:
    launches = 0

    def __init__(self, cp: _FakeCuPy) -> None:
        self.cp = cp
        self.metadata = {
            "optix_version": "9.1.0",
            "cuda_device": "synthetic GPU",
            "compute_capability": "89",
            "cuda_runtime_version": 12040,
            "nvrtc_version": (12, 4),
            "optix_include": "/synthetic/optix",
            "cuda_include": "/synthetic/cuda",
        }

    def build_gas(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> tuple[int, list[object]]:
        assert vertices.shape == (3, 3)
        assert faces.shape == (1, 3)
        return 17, []

    def launch(self, params_host: np.ndarray, **_: object) -> None:
        type(self).launches += 1
        result = self.cp.arrays[int(params_host["result"][0])]
        if float(params_host["origin"][0][0]) == 0.0:
            result.values[:] = (
                1,
                0,
                np.asarray(1.0, dtype=np.float32).view(np.uint32),
            )
        else:
            result.values[:] = 0


def _fake_import_modules(monkeypatch, fake_cp: _FakeCuPy) -> None:
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    monkeypatch.setitem(sys.modules, "optix", ModuleType("optix"))
    cuda = ModuleType("cuda")
    bindings = ModuleType("cuda.bindings")
    nvrtc = ModuleType("cuda.bindings.nvrtc")
    bindings.nvrtc = nvrtc
    cuda.bindings = bindings
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.nvrtc", nvrtc)


def _result() -> smoke.OptixSmokeResult:
    return smoke.OptixSmokeResult(
        metadata={
            "cuda_device": "synthetic GPU",
            "optix_version": "9.1.0",
            "cuda_runtime_version": 12040,
            "nvrtc_version": (12, 4),
            "optix_include": "/optix",
            "cuda_include": "/cuda",
        },
        hit=(1, 0, 1.0),
        miss=(0, 0, 0.0),
        setup_time_seconds=0.1,
        trace_time_seconds=0.01,
        ray_count=2,
        terminal_event_counts={"hit": 1, "miss": 1},
        result_counts={"hit": 1, "miss": 1},
    )


def test_smoke_success_performs_two_launches_without_fea_or_ax(
    monkeypatch,
) -> None:
    fake_cp = _FakeCuPy()
    _FakeRuntime.launches = 0
    _fake_import_modules(monkeypatch, fake_cp)
    monkeypatch.setattr(
        smoke.OptixRuntime,
        "create",
        lambda **_: _FakeRuntime(fake_cp),
    )

    result = smoke.run()

    assert _FakeRuntime.launches == 2
    assert result.hit == (1, 0, 1.0)
    assert result.miss == (0, 0, 0.0)
    source = inspect.getsource(smoke)
    assert "fem" not in source.lower()
    assert "AxClient" not in source
    assert "evaluationregistry" not in source.lower()


def test_cli_uses_shared_smoke_and_reports_failure_stage(monkeypatch, capsys) -> None:
    calls: list[str] = []

    def fail() -> smoke.OptixSmokeResult:
        calls.append("smoke")
        raise smoke.OptixSmokeError("nvrtc_compile", "synthetic NVRTC failure")

    monkeypatch.setattr(smoke, "run", fail)

    assert smoke.main() == 1
    assert calls == ["smoke"]
    assert "FAIL: nvrtc_compile" in capsys.readouterr().err


def test_cli_success_summary_contains_runtime_and_result_counts(monkeypatch, capsys) -> None:
    monkeypatch.setattr(smoke, "run", _result)

    assert smoke.main() == 0
    output = capsys.readouterr().out
    assert "PASS: optix_smoke" in output
    assert "setup=0.100s" in output
    assert "trace=0.010s" in output
    assert "rays=2" in output
    assert "terminal=hit=1,miss=1" in output
    assert "results=hit=1,miss=1" in output

"""Bounded subprocess isolation for scientific validation cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import resource
import subprocess
import time
from typing import Mapping, Sequence


@dataclass(frozen=True)
class IsolatedRun:
    """Terminal status of one bounded child process."""

    command: tuple[str, ...]
    return_code: int
    elapsed_seconds: float
    timed_out: bool

    @property
    def passed(self) -> bool:
        return not self.timed_out and self.return_code == 0


def run_isolated(
    command: Sequence[str],
    *,
    cwd: str | Path,
    timeout_seconds: float,
    environment: Mapping[str, str] | None = None,
    output_path: str | Path | None = None,
    disable_core_dumps: bool = False,
) -> IsolatedRun:
    """Run one child with an explicit timeout and no hidden retries."""
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive")
    argv = tuple(str(item) for item in command)
    start = time.perf_counter()
    output_stream = None
    try:
        if output_path is not None:
            resolved_output = Path(output_path)
            resolved_output.parent.mkdir(parents=True, exist_ok=True)
            output_stream = resolved_output.open("w", encoding="utf-8")
        completed = subprocess.run(
            argv,
            cwd=Path(cwd),
            env=None if environment is None else dict(environment),
            stdout=output_stream,
            stderr=subprocess.STDOUT if output_stream is not None else None,
            check=False,
            timeout=timeout_seconds,
            preexec_fn=(
                lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            )
            if disable_core_dumps
            else None,
        )
        return IsolatedRun(
            command=argv,
            return_code=int(completed.returncode),
            elapsed_seconds=time.perf_counter() - start,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        return IsolatedRun(
            command=argv,
            return_code=124,
            elapsed_seconds=time.perf_counter() - start,
            timed_out=True,
        )
    finally:
        if output_stream is not None:
            output_stream.close()

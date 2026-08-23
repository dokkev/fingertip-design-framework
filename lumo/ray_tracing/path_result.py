"""Result data produced by bounded optical path tracing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_ESCAPED_RAY_DTYPE = np.dtype(
    [
        ("ray_id", np.int64),
        ("bounce", np.int64),
        ("origin_W_m", np.float64, (3,)),
        ("direction_W", np.float64, (3,)),
        ("power", np.float64),
    ]
)

_PATH_SEGMENT_DTYPE = np.dtype(
    [
        ("ray_id", np.int64),
        ("bounce", np.int64),
        ("start_W_m", np.float64, (3,)),
        ("end_W_m", np.float64, (3,)),
        ("power", np.float64),
        ("instance_id", np.int32),
    ]
)


@dataclass(frozen=True)
class PathTraceResult:
    """Escaped paths, optional diagnostics, and scalar power accounting."""

    escaped_rays: np.ndarray
    emitted_power: float
    escaped_power: float
    absorbed_power: float
    bulk_loss_power: float
    unresolved_internal_miss_power: float
    remaining_power: float
    remaining_ray_count: int
    segments: np.ndarray | None = None

    @property
    def accounted_power(self) -> float:
        """Return the complete modeled power ledger."""
        return (
            self.escaped_power
            + self.absorbed_power
            + self.bulk_loss_power
            + self.unresolved_internal_miss_power
            + self.remaining_power
        )

    @property
    def closure_error(self) -> float:
        """Return accounted minus emitted power."""
        return self.accounted_power - self.emitted_power

    @property
    def escaped_ray_count(self) -> int:
        """Return the number of escaped path records."""
        return len(self.escaped_rays)


__all__ = ["PathTraceResult"]

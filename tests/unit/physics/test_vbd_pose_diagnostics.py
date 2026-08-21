"""Numerical contracts at the Warp prescribed-pose boundary."""

from __future__ import annotations

import numpy as np
import pytest

from lumo.mesh.rigid.object import RigidPose3D
from lumo.physics.newton.vbd import (
    _prescribed_pose_error_mm,
    _prescribed_pose_quantization_error_mm,
    _represented_translation_mm,
    _warp_translation_m,
)


@pytest.mark.parametrize(
    ("target_mm", "legacy_error_mm"),
    (
        ((11.028906358462883, -15.668373446345992, 0.0), 0.0),
        (
            (15.347312091668439, -19.314155955835925, 0.0),
            2.1324806311895372e-6,
        ),
        (
            (13.633614237689256, -18.272754604007417, 0.0),
            2.1324806311895372e-6,
        ),
    ),
)
def test_corrected_metric_removes_the_two_observed_rounding_failures(
    target_mm: tuple[float, float, float],
    legacy_error_mm: float,
) -> None:
    pose = RigidPose3D(target_mm, (0.0, 0.0, 0.0, 1.0))
    submitted_m = _warp_translation_m(pose)
    legacy_target_mm = np.asarray(target_mm, dtype=np.float32)
    legacy_actual_mm = submitted_m * np.float32(1.0e3)

    assert float(np.linalg.norm(legacy_actual_mm - legacy_target_mm)) == pytest.approx(
        legacy_error_mm,
        abs=1.0e-15,
    )
    assert _prescribed_pose_error_mm(submitted_m, submitted_m.copy()) == 0.0
    assert _prescribed_pose_quantization_error_mm(pose) >= 0.0


def test_pose_diagnostic_compares_with_the_float32_target_sent_to_warp() -> None:
    pose = RigidPose3D(
        (-22.158945083618164, -25.1242618560791, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    submitted_m = _warp_translation_m(pose)
    represented_mm = _represented_translation_mm(pose)

    assert submitted_m.dtype == np.float32
    assert represented_mm.dtype == np.float64
    np.testing.assert_array_equal(
        represented_mm,
        submitted_m.astype(np.float64) * 1.0e3,
    )
    assert np.linalg.norm(
        represented_mm
        - np.asarray(pose.translation_mm, dtype=np.float32).astype(np.float64)
    ) > 1.0e-6
    assert _prescribed_pose_error_mm(submitted_m, submitted_m.copy()) == 0.0
    assert _prescribed_pose_quantization_error_mm(pose) > 0.0


def test_pose_metric_detects_real_solver_deviation_without_quantization() -> None:
    target_m = np.asarray((0.011, -0.019, 0.0), dtype=np.float32)
    actual_m = target_m.copy()
    actual_m[1] += np.float32(2.0e-9)

    error_mm = _prescribed_pose_error_mm(actual_m, target_m)

    assert error_mm > 1.0e-6
    assert error_mm == float(
        np.linalg.norm(actual_m.astype(np.float64) - target_m.astype(np.float64))
        * 1.0e3
    )

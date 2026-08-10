from __future__ import annotations

import inspect
import subprocess
import sys

import numpy as np
import pytest

from model import Fingertip, FingertipParameters
from optics import TraceSettings, evaluate, trace
from optics.adapters import (
    OpticalFieldAdapterError,
    build_pad_mesh_from_arrays,
    load_pad_mesh_npz,
)


def test_evaluate_is_camera_independent_and_zero_for_identical_results() -> None:
    tip = Fingertip(FingertipParameters())
    settings = TraceSettings(
        ray_count=21,
        grid_width=32,
        grid_height=32,
        maximum_segment_count=3000,
    )
    reference = trace(tip, settings=settings)

    identical = evaluate(reference, reference)

    assert identical["field_difference"] == pytest.approx(0.0)
    assert identical["centroid_shift_mm"] == pytest.approx(0.0)
    assert identical["escaped_fraction_change"] == pytest.approx(0.0)
    assert identical["absorbed_fraction_change"] == pytest.approx(0.0)

    mesh = tip.mesh()
    displacement = np.zeros_like(mesh.coordinates)
    displacement[mesh.boundary_node_indices_for("pad_cutout_bottom"), 1] = -0.05
    loaded = trace(tip, mesh.deformed(displacement), settings)
    metrics = evaluate(reference, loaded)
    assert set(metrics) == {
        "field_difference",
        "centroid_shift_mm",
        "escaped_fraction_change",
        "absorbed_fraction_change",
    }
    assert all(np.isfinite(value) for value in metrics.values())


def test_external_adapters_require_no_legacy_boundary_classifier() -> None:
    assert "tip" not in inspect.signature(build_pad_mesh_from_arrays).parameters
    assert "tip" not in inspect.signature(load_pad_mesh_npz).parameters
    with pytest.raises(OpticalFieldAdapterError, match="semantic boundary"):
        build_pad_mesh_from_arrays(
            node_ids=np.asarray([1, 2, 3]),
            reference_coordinates_mm=np.asarray(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
            ),
            element_connectivity_node_ids=np.asarray([[1, 2, 3]]),
            displacement_mm=np.zeros((3, 2)),
        )


def test_core_imports_do_not_load_mitsuba() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import model, mesh, fem, optics; "
                "assert not any(name == 'mitsuba' or "
                "name.startswith('mitsuba.') for name in sys.modules); "
                "assert 'optics.mitsuba' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

from __future__ import annotations

import inspect
import subprocess
import sys

import numpy as np
import pytest

from model import Fingertip, FingertipParameters, LED
from optics import TraceSettings, evaluate, field_difference, trace
from optics.adapters import (
    OpticalFieldAdapterError,
    build_pad_mesh_from_arrays,
    load_pad_mesh_npz,
)


def test_evaluate_is_camera_independent_and_zero_for_identical_results() -> None:
    settings = TraceSettings(
        ray_count=21,
        grid_width=32,
        grid_height=32,
        maximum_segment_count=3000,
    )
    reference = trace(
        Fingertip(
            FingertipParameters(),
            led=LED(emission_half_angle_deg=60.0),
        ),
        settings=settings,
    )

    identical = evaluate(reference, reference)

    assert identical["field_difference"] == pytest.approx(0.0)
    assert field_difference(reference, reference) == pytest.approx(0.0)
    assert identical["centroid_shift_mm"] == pytest.approx(0.0)
    assert identical["escaped_fraction_change"] == pytest.approx(0.0)
    assert identical["absorbed_fraction_change"] == pytest.approx(0.0)

    loaded = trace(
        Fingertip(
            FingertipParameters(),
            led=LED(emission_half_angle_deg=80.0),
        ),
        settings=settings,
    )
    metrics = evaluate(reference, loaded)
    assert field_difference(reference, loaded) == pytest.approx(
        field_difference(loaded, reference)
    )
    assert metrics["field_difference"] == pytest.approx(
        field_difference(reference, loaded)
    )
    assert set(metrics) == {
        "field_difference",
        "centroid_shift_mm",
        "escaped_fraction_change",
        "absorbed_fraction_change",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["field_difference"] > 0.0
    assert metrics["centroid_shift_mm"] > 0.0


def test_new_npz_schema_round_trips_semantic_boundaries(tmp_path) -> None:
    node_ids = np.asarray([30, 10, 40, 20], dtype=np.int64)
    coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )
    connectivity = np.asarray(
        [[30, 40, 10], [30, 20, 40]],
        dtype=np.int64,
    )
    displacement = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [0.1, 0.2], [0.0, 0.2]],
        dtype=float,
    )
    path = tmp_path / "pad.npz"
    np.savez(
        path,
        node_ids=node_ids,
        reference_coordinates_mm=coordinates,
        element_connectivity_node_ids=connectivity,
        displacement_mm=displacement,
        boundary_edge_node_ids__bottom=np.asarray([[30, 10]], dtype=np.int64),
        boundary_edge_node_ids__right=np.asarray([[10, 40]], dtype=np.int64),
        boundary_edge_node_ids__top=np.asarray([[40, 20]], dtype=np.int64),
        boundary_edge_node_ids__left=np.asarray([[20, 30]], dtype=np.int64),
    )

    loaded = load_pad_mesh_npz(path)

    np.testing.assert_allclose(loaded.coordinates, coordinates + displacement)
    np.testing.assert_array_equal(loaded.node_ids, node_ids)
    assert loaded.semantic_boundary_tags == ("bottom", "left", "right", "top")
    np.testing.assert_array_equal(loaded.boundary_edges_for("bottom"), [[0, 1]])
    np.testing.assert_array_equal(loaded.boundary_edges_for("left"), [[3, 0]])
    np.testing.assert_array_equal(loaded.boundary_edges_for("right"), [[1, 2]])
    np.testing.assert_array_equal(loaded.boundary_edges_for("top"), [[2, 3]])


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
                "assert 'optics.mitsuba' not in sys.modules; "
                "assert 'cupy' not in sys.modules; "
                "assert 'optix' not in sys.modules; "
                "assert 'cuda' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

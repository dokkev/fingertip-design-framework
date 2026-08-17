from __future__ import annotations

import json

import numpy as np
import pytest

from model import Fingertip, FingertipParameters
from validation.optics import side_exit_flux_2d3d_pilot as pilot


def _elements_2d() -> list[dict[str, object]]:
    return [
        {
            "element_id": 0,
            "semantic_tag": "pad_outer_arc",
            "reference_centroid_mm": [0.0, -1.0],
            "reference_normal": [0.0, -1.0],
            "reference_measure": 2.0,
            "eligible_external": True,
        },
        {
            "element_id": 1,
            "semantic_tag": "pad_outer_left",
            "reference_centroid_mm": [-1.0, -0.5],
            "reference_normal": [-1.0, 0.0],
            "reference_measure": 3.0,
            "eligible_external": True,
        },
        {
            "element_id": 2,
            "semantic_tag": "pad_bond_left",
            "reference_centroid_mm": [-1.0, 0.0],
            "reference_normal": [0.0, 1.0],
            "reference_measure": 4.0,
            "eligible_external": False,
        },
        {
            "element_id": 3,
            "semantic_tag": "void_left",
            "reference_centroid_mm": [-0.5, -0.5],
            "reference_normal": [1.0, 0.0],
            "reference_measure": 5.0,
            "eligible_external": False,
        },
    ]


def _elements_3d() -> list[dict[str, object]]:
    return [
        {
            "element_id": 0,
            "semantic_tag": "outer_compliant_arc",
            "reference_centroid_mm": [0.0, -1.0, 0.0],
            "reference_normal": [0.0, -1.0, 0.0],
            "reference_measure": 2.0,
            "eligible_external": True,
        },
        {
            "element_id": 1,
            "semantic_tag": "outer_compliant_left",
            "reference_centroid_mm": [-1.0, -0.5, 0.0],
            "reference_normal": [-1.0, 0.0, 0.0],
            "reference_measure": 3.0,
            "eligible_external": True,
        },
        {
            "element_id": 2,
            "semantic_tag": "support_bond_left",
            "reference_centroid_mm": [-1.0, 0.0, 0.0],
            "reference_normal": [0.0, 1.0, 0.0],
            "reference_measure": 4.0,
            "eligible_external": False,
        },
        {
            "element_id": 3,
            "semantic_tag": "void_left",
            "reference_centroid_mm": [-0.5, -0.5, 0.0],
            "reference_normal": [1.0, 0.0, 0.0],
            "reference_measure": 5.0,
            "eligible_external": False,
        },
        {
            "element_id": 4,
            "semantic_tag": "longitudinal_end_minus",
            "reference_centroid_mm": [0.0, -0.5, -5.5],
            "reference_normal": [0.0, 0.0, -1.0],
            "reference_measure": 6.0,
            "eligible_external": False,
        },
    ]


def _payload(weights: dict[int, float], launched: float = 1.0) -> dict[str, object]:
    elements = _elements_2d()
    for element in elements:
        element["exit_weight_sum"] = weights.get(int(element["element_id"]), 0.0)
    escaped = float(sum(weights.values()))
    return {
        "reference_surface_elements": elements,
        "result": {
            "launched_weight": launched,
            "escaped_weight": escaped,
        },
    }


def test_2d_and_3d_exit_events_map_to_reference_elements_without_measure_multiplier() -> None:
    planar = pilot._aggregate_exit_events(
        _elements_2d(),
        {0: 0, 1: 0, 2: 1},
        [0, 1, 2],
        ["pad_outer_arc", "pad_outer_arc", "pad_outer_left"],
        [2.0, 3.0, 1.0],
        6.0,
    )
    assert planar["element_weights"].tolist()[:2] == [5.0, 1.0]
    assert planar["element_counts"].tolist()[:2] == [2, 1]
    assert planar["phi_external"] == pytest.approx(6.0)
    envelope_accounting = pilot._aggregate_exit_events(
        _elements_2d(),
        {0: 0},
        [0],
        ["pad_outer_arc"],
        [6.0],
        8.0,
        outgoing_surface_weight=6.0,
    )
    assert envelope_accounting["excluded_escape_categories"]["virtual_envelope"] == pytest.approx(2.0)

    spatial = pilot._aggregate_exit_events(
        _elements_3d(),
        {0: 0, 1: 1},
        [0, 1],
        ["outer_compliant_arc", "outer_compliant_left"],
        [2.0, 3.0],
        5.0,
    )
    assert spatial["element_weights"].tolist()[:2] == [2.0, 3.0]
    assert spatial["phi_external"] == pytest.approx(5.0)


@pytest.mark.parametrize("dimension,elements,primitive_to_element,tags", [
    ("2D", _elements_2d(), {0: 2}, ["pad_bond_left"]),
    ("2D", _elements_2d(), {0: 3}, ["void_left"]),
    ("3D", _elements_3d(), {0: 2}, ["support_bond_left"]),
    ("3D", _elements_3d(), {0: 3}, ["void_left"]),
    ("3D", _elements_3d(), {0: 4}, ["longitudinal_end_minus"]),
])
def test_ineligible_bond_void_and_cap_events_fail_closed(
    dimension: str,
    elements: list[dict[str, object]],
    primitive_to_element: dict[int, int],
    tags: list[str],
) -> None:
    del dimension
    with pytest.raises(ValueError, match="ineligible"):
        pilot._aggregate_exit_events(elements, primitive_to_element, [0], tags, [1.0], 1.0)


def test_reference_normals_and_threshold_partitions_are_deterministic() -> None:
    elements = _elements_2d()
    elements[0]["deformed_normal"] = [1.0, 0.0]
    first = [pilot._partition(elements, np.asarray([0.0, -1.0]), theta) for theta in pilot.THETAS_DEG]
    second = [pilot._partition(elements, np.asarray([0.0, -1.0]), theta) for theta in pilot.THETAS_DEG]
    assert first == second
    assert all(partition["front_element_ids"] == [0] for partition in first)
    assert all(partition["side_element_ids"] == [1] for partition in first)

    spatial = _elements_3d()
    partitions = [pilot._partition(spatial, np.asarray([0.0, -1.0, 0.0]), theta) for theta in pilot.THETAS_DEG]
    assert all(partition["front_element_ids"] == [0] for partition in partitions)
    assert all(partition["side_element_ids"] == [1] for partition in partitions)


def test_delta_eta_normalization_and_added_side_weight() -> None:
    partition = pilot._partition(_elements_2d(), np.asarray([0.0, -1.0]), 45.0)
    no_load = _payload({0: 0.4})
    loaded = _payload({0: 0.4, 1: 0.2})
    baseline = pilot._metrics(no_load, 45.0, partition, baseline=None)
    result = pilot._metrics(loaded, 45.0, partition, baseline=baseline)
    assert baseline["eta_side"] == pytest.approx(0.0)
    assert result["eta_side"] == pytest.approx(0.2)
    assert result["Delta_eta_side"] == pytest.approx(0.2)
    assert result["f_side"] == pytest.approx(1.0 / 3.0)

    normalized = pilot._metrics(_payload({1: 0.4}, launched=2.0), 45.0, partition, baseline=None)
    assert normalized["eta_side"] == pytest.approx(0.2)


def test_contract_preserves_native_dimension_and_load_provenance(tmp_path) -> None:
    tip = Fingertip(FingertipParameters())
    prepared = {
        "tip": tip,
        "mechanics_source": "NO_LOAD_REFERENCE_GEOMETRY",
        "mechanics_fingerprint": "NO_LOAD_REFERENCE",
        "reference_geometry_checksum": "reference",
        "elements": _elements_2d(),
    }
    settings = pilot._settings("2D", ((-20.0, 20.0), (-20.0, 2.0)))
    morphology = {"morphology_id": "m", "morphology_fingerprint": "m-fp"}
    planar = pilot._trace_contract(morphology, "2D", "no_load", prepared, settings)
    loaded = pilot._trace_contract(morphology, "2D", "left", {**prepared, "mechanics_source": "state.npz", "mechanics_fingerprint": "state"}, settings)
    spatial = pilot._trace_contract(morphology, "3D", "no_load", prepared, pilot._settings("3D", ((-20.0, 20.0), (-20.0, 2.0))))
    assert planar["optical_mode"] == "PLANAR_2D"
    assert planar["mechanics_dimension"] == "2D"
    assert spatial["optical_mode"] == "FULL_3D"
    assert spatial["mechanics_dimension"] == "3D"
    assert planar["load_state"] == "no_load"
    assert loaded["load_state"] == "left"
    assert loaded["mechanics_fingerprint"] != planar["mechanics_fingerprint"]

    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({"contract": {"morphology_fingerprint": "wrong"}}), encoding="utf-8")
    assert pilot._valid_trace(stale, planar) is None

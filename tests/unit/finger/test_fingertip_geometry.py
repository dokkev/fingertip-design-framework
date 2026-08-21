"""Tests for the canonical full-width link and recessed L-bond geometry."""

from __future__ import annotations

import math

import pytest
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box
from shapely.ops import linemerge, unary_union

from lumo.finger.fingertip_geometry import (
    FingertipModel,
    InvalidFingertipGeometry,
)
from lumo.finger.fingertip_parameters import (
    FingertipParameters,
    InvalidFingertipParameters,
    fingertip_parameters_fingerprint,
)


SIDE_CLEARANCE = 1.0
BOTTOM_CLEARANCE = 2.0


def build_model(**overrides: float | int) -> FingertipModel:
    return FingertipModel(FingertipParameters(**overrides))


def test_physical_morphology_fingerprint_ignores_representation() -> None:
    base = FingertipParameters()
    assert fingertip_parameters_fingerprint(base) == fingertip_parameters_fingerprint(
        FingertipParameters(
            arc_resolution=16,
            geometry_length_tolerance_mm=1.0e-8,
        )
    )
    assert fingertip_parameters_fingerprint(base) != fingertip_parameters_fingerprint(
        FingertipParameters(stem_height=6.1)
    )


def test_complete_geometry_is_symmetric_about_vertical_axis() -> None:
    model = build_model(void_width=SIDE_CLEARANCE, void_height=BOTTOM_CLEARANCE)
    mirrored = affinity.scale(
        model.material_geometry,
        xfact=-1.0,
        yfact=1.0,
        origin=(0.0, 0.0),
    )
    assert model.material_geometry.symmetric_difference(mirrored).area <= (
        model.parameters.geometry_area_tolerance_mm2
    )


def test_flat_pad_occupies_the_expected_rectangle() -> None:
    model = build_model()
    parameters = model.parameters
    expected = box(
        -parameters.flat_pad_width / 2.0,
        -parameters.flat_pad_height,
        parameters.flat_pad_width / 2.0,
        0.0,
    )
    assert model.flat_pad_geometry.equals(expected)


def test_outer_pad_unions_flat_pad_ellipse_and_recess_extensions() -> None:
    model = build_model(arc_resolution=17)
    parameters = model.parameters
    assert model.outer_pad_geometry.equals(
        model.flat_pad_geometry.union(model.semielliptical_pad_geometry)
        .union(model.left_bond_extension_geometry)
        .union(model.right_bond_extension_geometry)
    )
    assert model.outer_pad_geometry.bounds == pytest.approx(
        (
            -parameters.flat_pad_width / 2.0,
            -(parameters.flat_pad_height + parameters.semielliptical_pad_height),
            parameters.flat_pad_width / 2.0,
            parameters.bond_extension_height,
        )
    )
    assert model.outer_pad_geometry.is_valid
    assert isinstance(model.outer_pad_geometry, Polygon)


def test_semiellipse_meets_flat_pad_at_exact_endpoints() -> None:
    model = build_model()
    parameters = model.parameters
    ellipse_start_y = -parameters.flat_pad_height
    left = (-parameters.flat_pad_width / 2.0, ellipse_start_y)
    right = (parameters.flat_pad_width / 2.0, ellipse_start_y)
    arc_endpoints = (
        tuple(model.boundaries.pad_outer_arc.geometry.coords)[0],
        tuple(model.boundaries.pad_outer_arc.geometry.coords)[-1],
    )
    assert arc_endpoints[0] == pytest.approx(right)
    assert arc_endpoints[1] == pytest.approx(left)
    assert model.semielliptical_pad_geometry.area == pytest.approx(
        0.5
        * math.pi
        * (parameters.flat_pad_width / 2.0)
        * parameters.semielliptical_pad_height,
        rel=0.01,
    )


def test_bond_extensions_occupy_the_two_symmetric_recesses() -> None:
    model = build_model()
    parameters = model.parameters
    half_width = parameters.flat_pad_width / 2.0
    assert model.left_bond_extension_geometry.bounds == pytest.approx(
        (
            -half_width,
            0.0,
            -half_width + parameters.bond_extension_width,
            parameters.bond_extension_height,
        )
    )
    assert model.right_bond_extension_geometry.bounds == pytest.approx(
        (
            half_width - parameters.bond_extension_width,
            0.0,
            half_width,
            parameters.bond_extension_height,
        )
    )


def test_full_width_rigid_link_contains_two_recesses() -> None:
    model = build_model()
    parameters = model.parameters
    half_width = parameters.flat_pad_width / 2.0
    full_plate = box(
        -half_width,
        0.0,
        half_width,
        parameters.link_thickness,
    )
    left_recess = box(
        -half_width,
        0.0,
        -half_width + parameters.bond_extension_width,
        parameters.bond_extension_height,
    )
    right_recess = box(
        half_width - parameters.bond_extension_width,
        0.0,
        half_width,
        parameters.bond_extension_height,
    )
    expected_plate = full_plate.difference(left_recess.union(right_recess))
    assert model.link_plate_geometry.equals(expected_plate)
    assert model.link_plate_geometry.bounds == pytest.approx(
        (-half_width, 0.0, half_width, parameters.link_thickness)
    )


def test_cutout_is_top_open_and_contains_the_rigid_stem() -> None:
    model = build_model(void_width=SIDE_CLEARANCE, void_height=BOTTOM_CLEARANCE)
    parameters = model.parameters
    cutout_half_width = 0.5 * parameters.stem_width + parameters.void_width
    cutout_bottom_y = -(parameters.stem_height + parameters.void_height)
    expected_cutout = box(
        -cutout_half_width,
        cutout_bottom_y,
        cutout_half_width,
        0.0,
    )
    assert model.cutout_geometry.equals(expected_cutout)
    assert model.cutout_geometry.covers(model.stem_geometry)
    assert model.void_geometry is not None
    assert model.void_geometry.equals(
        model.cutout_geometry.difference(model.stem_geometry)
    )
    assert model.pad_material_geometry.equals(
        model.outer_pad_geometry.difference(model.cutout_geometry)
    )


def test_compliant_and_rigid_materials_do_not_overlap() -> None:
    model = build_model(void_width=SIDE_CLEARANCE, void_height=BOTTOM_CLEARANCE)
    assert model.pad_material_geometry.intersection(model.link_geometry).area == pytest.approx(
        0.0
    )
    assert model.material_geometry.is_valid
    assert model.is_material_connected()


def test_l_bond_boundaries_are_connected_symmetric_and_complete() -> None:
    model = build_model(void_width=SIDE_CLEARANCE)
    parameters = model.parameters
    half_width = parameters.flat_pad_width / 2.0
    cutout_half_width = 0.5 * parameters.stem_width + parameters.void_width
    left = model.boundaries.pad_bond_left.geometry
    right = model.boundaries.pad_bond_right.geometry
    assert isinstance(left, LineString)
    assert isinstance(right, LineString)
    expected_left = (
        (-cutout_half_width, 0.0),
        (-half_width + parameters.bond_extension_width, 0.0),
        (
            -half_width + parameters.bond_extension_width,
            parameters.bond_extension_height,
        ),
        (-half_width, parameters.bond_extension_height),
    )
    expected_right = (
        (half_width, parameters.bond_extension_height),
        (
            half_width - parameters.bond_extension_width,
            parameters.bond_extension_height,
        ),
        (half_width - parameters.bond_extension_width, 0.0),
        (cutout_half_width, 0.0),
    )
    for actual, expected in zip(left.coords, expected_left):
        assert actual == pytest.approx(expected)
    for actual, expected in zip(right.coords, expected_right):
        assert actual == pytest.approx(expected)
    assert model.pad_link_connection_length() == pytest.approx(
        parameters.flat_pad_width
        - 2.0 * cutout_half_width
        + 2.0 * parameters.bond_extension_height
    )
    assert isinstance(model.pad_link_interface, MultiLineString)
    assert len(model.pad_link_interface.geoms) == 2


def test_external_pad_shell_remains_one_open_connected_chain() -> None:
    model = build_model()
    segments = [
        model.boundaries.segments[tag].geometry
        for tag in (
            "pad_bond_left",
            "pad_outer_left",
            "pad_outer_arc",
            "pad_outer_right",
            "pad_bond_right",
        )
    ]
    shell = linemerge(unary_union(segments))
    assert isinstance(shell, LineString)
    assert shell.is_simple
    assert not shell.is_ring
    assert tuple(shell.coords)[0] == pytest.approx(
        (
            -(0.5 * model.parameters.stem_width + model.parameters.void_width),
            0.0,
        )
    )
    assert tuple(shell.coords)[-1] == pytest.approx(
        (0.5 * model.parameters.stem_width + model.parameters.void_width, 0.0)
    )


@pytest.mark.parametrize("stem_height", [2.0, 6.0])
def test_stem_height_does_not_change_outer_envelope(stem_height: float) -> None:
    reference = build_model(stem_height=4.0)
    changed = build_model(stem_height=stem_height)
    assert changed.outer_pad_geometry.equals(reference.outer_pad_geometry)
    assert changed.outer_pad_geometry.area == pytest.approx(
        reference.outer_pad_geometry.area
    )
    assert changed.stem_geometry.bounds[1] == pytest.approx(-stem_height)


@pytest.mark.parametrize(
    "clearance",
    [
        {"void_width": 1.0},
        {"void_height": 2.0},
        {"void_width": 1.0, "void_height": 2.0},
    ],
)
def test_void_dimensions_do_not_change_outer_envelope(
    clearance: dict[str, float],
) -> None:
    reference = build_model()
    changed = build_model(**clearance)
    assert changed.outer_pad_geometry.equals(reference.outer_pad_geometry)


def test_cutout_exiting_completed_outer_envelope_is_rejected() -> None:
    with pytest.raises(InvalidFingertipParameters, match="semielliptical"):
        build_model(stem_height=20.0)


def test_model_retains_defensive_cutout_containment_check(monkeypatch) -> None:
    monkeypatch.setattr(FingertipParameters, "validate", lambda self: None)
    parameters = FingertipParameters(stem_height=20.0)
    with pytest.raises(InvalidFingertipGeometry, match="cutout exits"):
        FingertipModel(parameters)


def test_rigid_link_area_reflects_recesses_and_stem() -> None:
    model = build_model()
    parameters = model.parameters
    expected_area = (
        parameters.flat_pad_width * parameters.link_thickness
        - 2.0
        * parameters.bond_extension_width
        * parameters.bond_extension_height
        + parameters.stem_width * parameters.stem_height
    )
    assert model.link_geometry.area == pytest.approx(expected_area)
    assert model.link_geometry.bounds == pytest.approx(
        (
            -parameters.flat_pad_width / 2.0,
            -parameters.stem_height,
            parameters.flat_pad_width / 2.0,
            parameters.link_thickness,
        )
    )


@pytest.mark.parametrize(
    ("overrides", "classification", "geometry_type"),
    [
        ({"void_width": 0.0}, "zero_clearance_fit", type(None)),
        ({"void_width": SIDE_CLEARANCE}, "side_clearance", MultiPolygon),
        (
            {"void_width": 0.0, "void_height": BOTTOM_CLEARANCE},
            "bottom_clearance",
            Polygon,
        ),
        (
            {"void_width": SIDE_CLEARANCE, "void_height": BOTTOM_CLEARANCE},
            "u_clearance",
            Polygon,
        ),
    ],
)
def test_limiting_clearance_cases(
    overrides: dict[str, float],
    classification: str,
    geometry_type: type[None] | type[Polygon] | type[MultiPolygon],
) -> None:
    model = build_model(**overrides)
    parameters = model.parameters
    cutout_width = parameters.stem_width + 2.0 * parameters.void_width
    cutout_height = parameters.stem_height + parameters.void_height
    expected_area = (
        cutout_width * cutout_height
        - parameters.stem_width * parameters.stem_height
    )
    assert model.classify_void() == classification
    assert isinstance(model.void_geometry, geometry_type)
    actual_area = 0.0 if model.void_geometry is None else model.void_geometry.area
    assert actual_area == pytest.approx(expected_area)
    assert model.interface_definition.interface_type == "bonded"


def test_material_area_decreases_only_by_visible_clearance() -> None:
    model = build_model(void_width=SIDE_CLEARANCE, void_height=BOTTOM_CLEARANCE)
    parameters = model.parameters
    void_area = (
        (parameters.stem_width + 2.0 * parameters.void_width)
        * (parameters.stem_height + parameters.void_height)
        - parameters.stem_width * parameters.stem_height
    )
    assert model.raw_material_geometry.area - model.material_geometry.area == pytest.approx(
        void_area
    )


def test_required_boundaries_and_contact_pairs_remain_explicit() -> None:
    model = build_model(void_width=SIDE_CLEARANCE, void_height=BOTTOM_CLEARANCE)
    required_tags = {
        "pad_bond_left",
        "pad_bond_right",
        "pad_outer_left",
        "pad_outer_right",
        "pad_cutout_left",
        "pad_cutout_right",
        "pad_cutout_bottom",
        "stem_left",
        "stem_right",
        "stem_bottom",
        "pad_outer_arc",
    }
    assert set(model.boundaries.segments) == required_tags
    assert all(
        not segment.geometry.is_empty
        for segment in model.boundaries.segments.values()
    )
    gaps = {pair.name: pair.initial_normal_gap for pair in model.contact_pairs}
    assert gaps == pytest.approx(
        {
            "left_contact": SIDE_CLEARANCE,
            "right_contact": SIDE_CLEARANCE,
            "bottom_contact": BOTTOM_CLEARANCE,
        }
    )


def test_zero_clearance_keeps_distinct_coincident_contact_boundaries() -> None:
    model = build_model(void_width=0.0)
    for pair in model.contact_pairs:
        assert pair.initial_normal_gap == pytest.approx(0.0)
        assert pair.stem_boundary is not pair.pad_boundary
        assert pair.stem_boundary.geometry.equals(pair.pad_boundary.geometry)
    assert model.void_geometry is None


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"void_width": SIDE_CLEARANCE},
        {"void_height": BOTTOM_CLEARANCE},
        {"void_width": SIDE_CLEARANCE, "void_height": BOTTOM_CLEARANCE},
    ],
)
def test_generated_domains_are_valid_and_connected(overrides: dict[str, float]) -> None:
    model = build_model(**overrides)
    assert model.outer_pad_geometry.is_valid
    assert model.pad_material_geometry.is_valid
    assert model.link_geometry.is_valid
    assert model.material_geometry.is_valid
    assert model.is_material_connected()
    model.validate_geometry()

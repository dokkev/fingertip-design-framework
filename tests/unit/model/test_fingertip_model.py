"""Tests for LIT pad construction and clearance geometry."""

from __future__ import annotations

import math

import pytest
from shapely import affinity
from shapely.geometry import MultiLineString, MultiPolygon, Polygon

from model.fingertip_model import FingertipModel
from model.fingertip_parameters import FingertipParameters


def build_model(**overrides: float | int | bool) -> FingertipModel:
    return FingertipModel(FingertipParameters(**overrides))


def test_complete_geometry_is_symmetric_about_vertical_axis() -> None:
    model = build_model(void_width=2.5, void_height=3.0)
    mirrored = affinity.scale(
        model.material_geometry,
        xfact=-1.0,
        yfact=1.0,
        origin=(0.0, 0.0),
    )
    mismatch_area = model.material_geometry.symmetric_difference(mirrored).area
    assert mismatch_area <= model.parameters.geometry_tolerance


def test_half_ellipse_uses_y_zero_interface_and_exact_depth() -> None:
    model = build_model(arc_resolution=17)
    assert model.outer_pad_geometry.bounds == pytest.approx((-15.0, -18.0, 15.0, 0.0))
    expected_area = math.pi * 15.0 * 18.0 / 2.0
    assert model.outer_pad_geometry.area == pytest.approx(expected_area, rel=0.01)


def test_cutout_corner_on_analytic_ellipse_boundary_is_accepted() -> None:
    base = FingertipParameters()
    boundary_depth = base.pad_height * math.sqrt(
        1.0 - (base.cutout_half_width / (base.pad_width / 2.0)) ** 2
    )
    model = build_model(stem_height=boundary_depth)
    assert model.parameters.cutout_depth == pytest.approx(boundary_depth)
    model.validate_geometry()


def test_rigid_link_contains_top_plate_and_inserted_stem() -> None:
    model = build_model()
    parameters = model.parameters
    expected_area = (
        parameters.link_width * parameters.link_thickness
        + parameters.stem_width * parameters.stem_height
    )
    assert model.link_geometry.area == pytest.approx(expected_area)
    assert model.link_geometry.bounds == pytest.approx(
        (
            -parameters.pad_width / 2.0,
            -parameters.stem_height,
            parameters.pad_width / 2.0,
            parameters.link_thickness,
        )
    )


@pytest.mark.parametrize(
    ("overrides", "classification", "geometry_type", "expected_area"),
    [
        ({}, "zero_clearance_fit", type(None), 0.0),
        (
            {"void_width": 2.5},
            "side_clearance",
            MultiPolygon,
            2.0 * 2.5 * 7.0,
        ),
        (
            {"void_height": 3.0},
            "bottom_clearance",
            Polygon,
            7.0 * 3.0,
        ),
        (
            {"void_width": 2.5, "void_height": 3.0},
            "u_clearance",
            Polygon,
            71.0,
        ),
    ],
)
def test_limiting_clearance_cases(
    overrides: dict[str, float],
    classification: str,
    geometry_type: type[None] | type[Polygon] | type[MultiPolygon],
    expected_area: float,
) -> None:
    model = build_model(**overrides)
    assert model.classify_void() == classification
    assert isinstance(model.void_geometry, geometry_type)
    actual_area = 0.0 if model.void_geometry is None else model.void_geometry.area
    assert actual_area == pytest.approx(expected_area)
    assert model.pad_link_connection_length() > 0.0
    assert model.interface_definition.interface_type == "bonded"


def test_pad_cutout_area_matches_full_cutout_rectangle() -> None:
    model = build_model(void_width=2.5, void_height=3.0)
    assert (
        model.outer_pad_geometry.area - model.pad_material_geometry.area
        == pytest.approx(model.parameters.cutout_width * model.parameters.cutout_height)
    )


def test_material_area_decreases_only_by_visible_clearance() -> None:
    model = build_model(void_width=2.5, void_height=3.0)
    assert (
        model.raw_material_geometry.area - model.material_geometry.area
        == pytest.approx(model.parameters.void_area)
    )


def test_interface_has_two_segments_outside_cutout() -> None:
    model = build_model(void_width=2.5)
    assert isinstance(model.pad_link_interface, MultiLineString)
    assert len(model.pad_link_interface.geoms) == 2
    assert model.pad_link_connection_length() == pytest.approx(
        model.parameters.pad_width - model.parameters.cutout_width
    )
    assert model.interface_definition.interface_type == "bonded"


def test_deprecated_bonded_flag_cannot_change_upper_interface() -> None:
    bonded = build_model(bonded=True, void_width=2.5, void_height=3.0)
    with pytest.warns(DeprecationWarning, match="always bonded"):
        legacy_unbonded = build_model(
            bonded=False,
            void_width=2.5,
            void_height=3.0,
        )
    assert bonded.material_geometry.equals(legacy_unbonded.material_geometry)
    assert bonded.interface_definition.interface_type == "bonded"
    assert legacy_unbonded.interface_definition.interface_type == "bonded"
    assert legacy_unbonded.pad_link_interface.equals(bonded.pad_link_interface)


def test_all_required_boundary_tags_are_explicit() -> None:
    model = build_model(void_width=2.5, void_height=3.0)
    required_tags = {
        "pad_bond_left",
        "pad_bond_right",
        "pad_cutout_left",
        "pad_cutout_right",
        "pad_cutout_bottom",
        "stem_left",
        "stem_right",
        "stem_bottom",
        "pad_outer_arc",
    }
    assert set(model.boundaries.segments) == required_tags
    assert all(not segment.geometry.is_empty for segment in model.boundaries.segments.values())


def test_contact_pairs_report_initial_normal_gaps() -> None:
    model = build_model(void_width=2.5, void_height=3.0)
    gaps = {pair.name: pair.initial_normal_gap for pair in model.contact_pairs}
    assert gaps == pytest.approx(
        {
            "left_contact": 2.5,
            "right_contact": 2.5,
            "bottom_contact": 3.0,
        }
    )
    assert model.contact_pairs[0].stem_boundary.name == "stem_left"
    assert model.contact_pairs[0].pad_boundary.name == "pad_cutout_left"
    for pair in model.contact_pairs:
        assert pair.stem_boundary.geometry.distance(
            pair.pad_boundary.geometry
        ) == pytest.approx(pair.initial_normal_gap)


def test_zero_clearance_keeps_distinct_coincident_contact_boundaries() -> None:
    model = build_model(void_width=0.0, void_height=0.0)
    for pair in model.contact_pairs:
        assert pair.initial_normal_gap == pytest.approx(0.0)
        assert pair.stem_boundary is not pair.pad_boundary
        assert pair.stem_boundary.geometry.equals(pair.pad_boundary.geometry)
    assert model.void_geometry is None


def test_zero_side_gap_keeps_coincident_side_tags_with_bottom_clearance() -> None:
    model = build_model(void_width=0.0, void_height=3.0)
    for pair in model.contact_pairs[:2]:
        assert pair.initial_normal_gap == pytest.approx(0.0)
        assert pair.stem_boundary.geometry.equals(pair.pad_boundary.geometry)


def test_zero_bottom_gap_keeps_coincident_bottom_tags_with_side_clearance() -> None:
    model = build_model(void_width=2.5, void_height=0.0)
    bottom_pair = model.contact_pairs[2]
    assert bottom_pair.initial_normal_gap == pytest.approx(0.0)
    assert bottom_pair.stem_boundary.geometry.equals(bottom_pair.pad_boundary.geometry)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"void_width": 2.5},
        {"void_height": 3.0},
        {"void_width": 2.5, "void_height": 3.0},
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


def test_summary_is_internally_consistent() -> None:
    model = build_model(void_width=2.5, void_height=3.0)
    summary = model.summary()
    assert summary["void_classification"] == "u_clearance"
    assert summary["cutout_width"] == pytest.approx(model.parameters.cutout_width)
    assert summary["cutout_height"] == pytest.approx(model.parameters.cutout_height)
    assert summary["void_area"] == pytest.approx(model.parameters.void_area)
    assert summary["material_area"] == pytest.approx(model.material_geometry.area)
    assert summary["removed_material_area"] == pytest.approx(summary["void_area"])
    assert summary["material_connected"] is True
    assert summary["geometry_valid"] is True
    assert set(summary["boundary_tags"]) == set(model.boundaries.segments)
    assert summary["contact_gaps"] == pytest.approx(
        {
            "left_contact": 2.5,
            "right_contact": 2.5,
            "bottom_contact": 3.0,
        }
    )

"""Integrity checks for the checked-in Figure 6(c) uncalibrated-location table.

The table is entered by hand rather than emitted by ``experiments.analysis``,
so these tests stand in for the provenance the generated artifacts carry.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from figures.fig6.fig6abc import (  # noqa: E402
    PANEL_C_MINIMUM_FIT_LOCATIONS,
    PANEL_C_REFERENCE,
    _panel_c_series,
    load_panel_c,
)


MATERIALS = ("solaris", "dragon_skin")
MORPHOLOGIES = ("baseline", "flat_opt", "angled_opt")
BUDGETS = tuple(range(6))


def test_table_covers_every_specimen_and_budget_exactly_once() -> None:
    rows = load_panel_c()
    keys = [
        (row["material"], row["morphology_id"], int(row["calibration_locations"]))
        for row in rows
    ]

    assert len(keys) == len(set(keys))
    assert set(keys) == {
        (material, morphology, budget)
        for material in MATERIALS
        for morphology in MORPHOLOGIES
        for budget in BUDGETS
    }


def test_every_row_is_either_measured_with_a_value_or_explained() -> None:
    for row in load_panel_c():
        if row["status"] == "measured":
            assert float(row["mae_mm_uncalibrated"]) > 0.0
            assert row["failure_reason"] == ""
        else:
            assert row["status"] == "unavailable"
            assert row["mae_mm_uncalibrated"] == ""
            assert row["failure_reason"] != ""


def test_single_calibration_location_never_carries_a_value() -> None:
    """The affine fit needs two locations, so K=1 must stay unavailable."""

    single = [
        row
        for row in load_panel_c()
        if int(row["calibration_locations"]) < PANEL_C_MINIMUM_FIT_LOCATIONS
        and int(row["calibration_locations"]) > 0
    ]

    assert len(single) == len(MATERIALS) * len(MORPHOLOGIES)
    assert all(row["status"] == "unavailable" for row in single)


def test_reference_line_specimen_has_a_prior() -> None:
    """The panel draws its dashed reference from this K=0 value."""

    material, morphology = PANEL_C_REFERENCE
    series = _panel_c_series(load_panel_c(), material, morphology)

    assert series[0][0] == 0


def test_dragon_skin_reports_no_geometry_prior() -> None:
    rows = load_panel_c()

    for morphology in MORPHOLOGIES:
        budgets = [point[0] for point in _panel_c_series(rows, "dragon_skin", morphology)]
        assert 0 not in budgets


def test_series_are_measured_only_and_ordered_by_calibration_effort() -> None:
    rows = load_panel_c()

    for material in MATERIALS:
        for morphology in MORPHOLOGIES:
            series = _panel_c_series(rows, material, morphology)
            budgets = [point[0] for point in series]
            assert budgets == sorted(budgets)
            assert 1 not in budgets
            assert all(point[1] > 0.0 for point in series)

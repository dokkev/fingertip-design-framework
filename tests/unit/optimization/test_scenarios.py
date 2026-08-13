from __future__ import annotations

import pytest

from optimization.scenarios import ContactScenario, ScenarioGrid


def test_contact_scenario_accepts_signed_finite_locations() -> None:
    assert ContactScenario(-2.0, 0.5, 2.0).location_x_mm == -2.0
    assert ContactScenario(0.0, 0.5, 2.0).location_x_mm == 0.0
    assert ContactScenario(2.0, 0.5, 2.0).location_x_mm == 2.0


@pytest.mark.parametrize(
    ("location", "indentation", "radius"),
    (
        (float("inf"), 0.5, 2.0),
        (0.0, float("inf"), 2.0),
        (0.0, 0.5, float("inf")),
    ),
)
def test_contact_scenario_rejects_nonfinite_values(
    location: float,
    indentation: float,
    radius: float,
) -> None:
    with pytest.raises(ValueError):
        ContactScenario(location, indentation, radius)


@pytest.mark.parametrize(
    ("indentation", "radius"),
    ((0.0, 2.0), (-1.0, 2.0), (0.5, 0.0), (0.5, -1.0)),
)
def test_contact_scenario_rejects_nonpositive_load_dimensions(
    indentation: float,
    radius: float,
) -> None:
    with pytest.raises(ValueError):
        ContactScenario(0.0, indentation, radius)


def test_grid_validates_nonempty_strictly_increasing_levels() -> None:
    grid = ScenarioGrid(
        locations_x_mm=(-1.0, 1.0),
        indentations_mm=(0.5, 1.0),
        indenter_radii_mm=(2.0, 4.0),
    )

    assert grid.locations_x_mm == (-1.0, 1.0)
    assert grid.indentations_mm == (0.5, 1.0)
    assert grid.indenter_radii_mm == (2.0, 4.0)
    assert [
        (scenario.location_x_mm, scenario.indentation_mm, scenario.indenter_radius_mm)
        for scenario in grid.scenarios
    ] == [
        (-1.0, 0.5, 2.0),
        (1.0, 0.5, 2.0),
        (-1.0, 1.0, 2.0),
        (1.0, 1.0, 2.0),
        (-1.0, 0.5, 4.0),
        (1.0, 0.5, 4.0),
        (-1.0, 1.0, 4.0),
        (1.0, 1.0, 4.0),
    ]


@pytest.mark.parametrize(
    "grid_values",
    (
        {"locations_x_mm": ()},
        {"locations_x_mm": (1.0, 0.0)},
        {"locations_x_mm": (0.0, 0.0)},
        {"indentations_mm": (1.0, 0.5)},
        {"indenter_radii_mm": (4.0, 2.0)},
    ),
)
def test_grid_rejects_empty_unsorted_or_duplicate_levels(grid_values) -> None:
    values = {
        "locations_x_mm": (-1.0, 1.0),
        "indentations_mm": (0.5, 1.0),
        "indenter_radii_mm": (2.0, 4.0),
    }
    values.update(grid_values)
    with pytest.raises(ValueError):
        ScenarioGrid(**values)


def test_grid_generates_only_deterministic_adjacent_pairs() -> None:
    grid = ScenarioGrid(
        locations_x_mm=(-1.0, 1.0),
        indentations_mm=(0.5, 1.0),
        indenter_radii_mm=(2.0, 4.0),
    )
    pairs = grid.adjacent_pairs

    assert len(pairs) == 12
    assert {pair.axis for pair in pairs} == {
        "location",
        "indentation",
        "radius",
    }
    assert sum(pair.axis == "location" for pair in pairs) == 4
    assert sum(pair.axis == "indentation" for pair in pairs) == 4
    assert sum(pair.axis == "radius" for pair in pairs) == 4
    assert len(
        {
            frozenset((pair.first, pair.second))
            for pair in pairs
        }
    ) == len(pairs)
    for pair in pairs:
        differences = (
            pair.first.location_x_mm != pair.second.location_x_mm,
            pair.first.indentation_mm != pair.second.indentation_mm,
            pair.first.indenter_radius_mm != pair.second.indenter_radius_mm,
        )
        assert sum(differences) == 1
    assert pairs == grid.adjacent_pairs

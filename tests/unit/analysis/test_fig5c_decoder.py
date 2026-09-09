from dataclasses import replace
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np

from experiments.analysis.fig5c_decoder import (
    DecoderSample,
    PaperFigureConfig,
    calibration_burden,
    decode_leave_one_repetition_out,
    regionize_profile_change,
)
from figures.fig6.fig6abc import _plot_panel_b


def _config(tmp_path: Path) -> PaperFigureConfig:
    return PaperFigureConfig(
        analysis_output_directory=tmp_path,
        figure5_output_directory=tmp_path,
        figure6_output_directory=tmp_path,
        profile_roots={"material": tmp_path},
        condition_overrides={},
        materials=("material",),
        indenters=("sphere",),
        morphologies=("baseline",),
        material_labels={"material": "Material"},
        indenter_labels={"sphere": "Sphere"},
        morphology_labels={"baseline": "Baseline"},
        morphology_colors={"baseline": "#777777"},
        contact_positions_mm={1: 0.0, 2: 10.0},
        low_force_n=2.0,
        high_force_n=5.0,
        region_count=2,
        maximum_calibration_contacts=3,
        maximum_resamples=100,
        random_seed=7,
    )


def _samples(tmp_path: Path) -> list[DecoderSample]:
    output = []
    for hole, center in ((1, 0.0), (2, 10.0)):
        for repetition, offset in enumerate((-0.2, 0.0, 0.1, 0.2), start=1):
            output.append(
                DecoderSample(
                    material="material",
                    indenter="sphere",
                    morphology="baseline",
                    specimen_id="specimen",
                    run_id=f"run_{hole}_{repetition}",
                    hole_index=hole,
                    repetition_index=repetition,
                    actual_force_low_n=2.0,
                    actual_force_high_n=5.0,
                    feature=np.asarray([center + offset, center - offset]),
                    source_path=tmp_path / "profiles.npz",
                )
            )
    return output


def test_regionize_profile_change_preserves_signed_regional_means() -> None:
    coordinate = np.linspace(0.0, 1.0, 13)
    low = np.zeros_like(coordinate)
    high = np.where(coordinate < 0.5, 2.0, -3.0)

    result = regionize_profile_change(low, high, coordinate, region_count=2)

    assert np.array_equal(result, np.asarray([2.0, -3.0]))


def test_leave_one_repetition_out_nearest_templates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    predictions = decode_leave_one_repetition_out(_samples(tmp_path), config)

    assert len(predictions) == 8
    assert all(row["spatial_correct"] for row in predictions)
    assert all(row["scalar_correct"] for row in predictions)


def test_calibration_burden_uses_only_unselected_repetitions(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rows = calibration_burden(_samples(tmp_path), config)

    assert [row["calibration_contacts_per_location"] for row in rows] == [1, 2, 3]
    assert [row["split_count"] for row in rows] == [4, 6, 4]
    assert all(row["test_accuracy_mean"] == 100.0 for row in rows)


def test_current_distinguishability_panel_marks_unavailable_morphology(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        morphologies=("baseline", "flat_opt", "angled_opt"),
        morphology_labels={
            "baseline": "Baseline",
            "flat_opt": "Progressive-constraint I",
            "angled_opt": "Progressive-constraint II",
        },
        morphology_colors={
            "baseline": "#AAAAAA",
            "flat_opt": "#555555",
            "angled_opt": "#222222",
        },
    )
    rows = [
        {
            "material": "material",
            "indenter": "sphere",
            "morphology_id": morphology,
            "Q_recontact": value,
            "status": status,
        }
        for morphology, value, status in (
            ("baseline", "2.0", "measured"),
            ("flat_opt", "3.0", "measured"),
            ("angled_opt", "", "unavailable"),
        )
    ]
    figure = Figure()
    axis = figure.subplots()

    _plot_panel_b(axis, rows, config)

    assert len(axis.patches) == 2
    assert [text.get_text() for text in axis.texts].count("n/a") == 1

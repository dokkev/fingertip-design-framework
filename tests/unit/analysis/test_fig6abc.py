from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from algorithm.contact_localization import (
    ContactLocalizationConfig,
    UnloadedOpticalReference,
    localize_response_profile,
)
from experiments.analysis.fig5c_decoder import PaperFigureConfig
import experiments.analysis.fig6abc as fig6abc
from experiments.analysis.fig6abc import (
    AnchorRegistration,
    ContactObservation,
    aggregate_panel_a,
    aggregate_run_first_cycle_values,
    detect_led_anchor_lattice,
    enumerate_repetition_splits,
    evaluate_optional_calibration,
    panel_b_values,
    physical_coordinates_from_led_rows,
    summarize_optional_calibration_predictions,
    validated_recontact_ratio,
)


def _config(
    tmp_path: Path,
    *,
    materials: tuple[str, ...] = ("material",),
    indenters: tuple[str, ...] = ("sphere_10mm",),
    morphologies: tuple[str, ...] = ("baseline",),
) -> PaperFigureConfig:
    return PaperFigureConfig(
        analysis_output_directory=tmp_path,
        figure5_output_directory=tmp_path,
        figure6_output_directory=tmp_path,
        profile_roots={material: tmp_path for material in materials},
        condition_overrides={},
        materials=materials,
        indenters=indenters,
        morphologies=morphologies,
        material_labels={material: material.title() for material in materials},
        indenter_labels={
            "sphere_10mm": "10 mm sphere",
            "sphere_30mm": "30 mm sphere",
        },
        morphology_labels={morphology: morphology for morphology in morphologies},
        morphology_colors={morphology: "#777777" for morphology in morphologies},
        contact_positions_mm={hole: 10.0 * (hole - 1) for hole in range(1, 7)},
        low_force_n=2.0,
        high_force_n=5.0,
        region_count=6,
        maximum_calibration_contacts=4,
        maximum_resamples=100,
        random_seed=7,
    )


def _observation(
    hole: int,
    repetition: int,
    *,
    geometry_valid: bool = True,
    optical_valid: bool = True,
) -> ContactObservation:
    coordinate = np.arange(fig6abc.PROFILE_BIN_COUNT, dtype=np.float64)
    center = 8.0 + 20.0 * (hole - 1)
    feature = np.exp(-0.5 * ((coordinate - center) / 2.0) ** 2)
    feature += 1.0e-3 * repetition
    position = 10.0 * (hole - 1) + 1.0
    return ContactObservation(
        observation_id=f"run_h{hole}_r{repetition}:5N",
        material="material",
        morphology="baseline",
        specimen_id="specimen",
        run_id=f"run_h{hole}_r{repetition}",
        hole_index=hole,
        repetition_index=repetition,
        calibration_index=0,
        reference_id="calibration_000",
        frame_count=2,
        actual_force_median_n=5.0,
        actual_force_std_n=0.05,
        actual_force_min_n=4.95,
        actual_force_max_n=5.05,
        geometry_valid=geometry_valid,
        contact_valid=optical_valid,
        geometry_status="valid" if geometry_valid else "invalid geometry",
        contact_status="valid" if optical_valid else "invalid optical evidence",
        response_profile=feature.copy(),
        evidence_profile=feature.copy(),
        geometry_position_mm=position if geometry_valid else None,
        geometry_localization_valid=geometry_valid,
        geometry_localization_status=(
            "valid" if geometry_valid else "invalid geometry"
        ),
    )


def _complete_observations() -> list[ContactObservation]:
    return [
        _observation(hole, repetition)
        for hole in range(1, 7)
        for repetition in range(1, 6)
    ]


def test_panel_a_reduces_branch_force_cells_within_run_first() -> None:
    rows = [
        {"run_id": "run_1", "W_cycle_dn": value}
        for value in (1.0, 9.0)
    ] + [{"run_id": "run_2", "W_cycle_dn": 3.0}]

    run_ids, run_values = aggregate_run_first_cycle_values(rows)

    assert run_ids == ("run_1", "run_2")
    np.testing.assert_allclose(run_values, (5.0, 3.0))
    assert float(np.median(run_values)) == 4.0
    assert float(np.median([1.0, 9.0, 3.0])) == 3.0


def test_panel_a_records_only_real_indenter_support(
    tmp_path: Path, monkeypatch
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "same_contact_repeatability.csv").write_text(
        "morphology,run_id,hole_index,actual_force_n,W_cycle_dn\n"
        "baseline,run_1,1,5,1\n"
        "baseline,run_1,1,6,9\n"
        "baseline,run_2,3,5,3\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(fig6abc.HISTORY_ROOTS, "material", tmp_path)
    monkeypatch.setattr(
        fig6abc,
        "_history_indenter_support",
        lambda _material, _morphology: ("sphere_10mm",),
    )
    monkeypatch.setattr(
        fig6abc,
        "_history_specimen_id",
        lambda _material, _morphology: "specimen",
    )
    config = _config(
        tmp_path,
        indenters=("sphere_10mm", "sphere_30mm"),
    )
    recontact = [
        {
            "material": "material",
            "indenter": "sphere_10mm",
            "morphology_id": "baseline",
            "W_recontact_DN_per_N": 0.25,
            "source_path": "metrics.csv",
            "status": "measured",
        }
    ]

    rows = aggregate_panel_a(config, recontact)

    assert len(rows) == 1
    assert rows[0]["indenter"] == "sphere_10mm"
    assert rows[0]["indenter_diameter_mm"] == 10.0
    assert rows[0]["W_cycle_DN"] == 4.0
    assert rows[0]["W_recontact_DN_per_N"] == 0.25
    assert rows[0]["source_run_count"] == 2
    assert rows[0]["status"] == "measured"


def test_panel_a_does_not_invent_missing_recontact_coordinate(
    tmp_path: Path, monkeypatch
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "same_contact_repeatability.csv").write_text(
        "morphology,run_id,hole_index,actual_force_n,W_cycle_dn\n"
        "baseline,run_1,1,5,1\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(fig6abc.HISTORY_ROOTS, "material", tmp_path)
    monkeypatch.setattr(
        fig6abc,
        "_history_indenter_support",
        lambda _material, _morphology: ("sphere_10mm",),
    )
    monkeypatch.setattr(
        fig6abc,
        "_history_specimen_id",
        lambda _material, _morphology: "specimen",
    )

    row = aggregate_panel_a(_config(tmp_path), [])[0]

    assert row["W_cycle_DN"] == 1.0
    assert row["W_recontact_DN_per_N"] == ""
    assert row["status"] == "unavailable"


def test_panel_b_uses_matching_aggregate_fields_and_not_w_cycle(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "morphology_metrics.csv").write_text(
        "material,morphology,indenter,D_neighbor_median_DN_per_N,"
        "W_median_DN_per_N,D_neighbor_over_W,W_cycle_dn\n"
        "material,baseline,sphere_30mm,99,1,99,0.001\n"
        "material,baseline,sphere_10mm,4,2,2,0.001\n",
        encoding="utf-8",
    )

    row = panel_b_values(_config(tmp_path))[0]

    assert row["D_neighbor_DN_per_N"] == 4.0
    assert row["W_recontact_DN_per_N"] == 2.0
    assert row["Q_recontact"] == 2.0
    assert row["denominator_source_field"] == "W_median_DN_per_N"
    assert row["status"] == "measured"


def test_panel_b_rejects_nonpositive_denominator_without_epsilon() -> None:
    ratio, reason = validated_recontact_ratio(4.0, 0.0, np.inf)

    assert ratio is None
    assert reason == "missing or nonpositive W_recontact denominator"


def test_led_mapping_preserves_distal_orientation_pitch_and_end_support() -> None:
    coordinates = np.arange(0.0, 61.0, 10.0)
    mapped = physical_coordinates_from_led_rows(
        coordinates,
        np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]),
    )

    np.testing.assert_allclose(mapped, (-11.0, 0.0, 11.0, 22.0, 33.0, 44.0, 55.0))


def test_unloaded_led_detector_returns_distal_to_proximal_five_tooth_lattice() -> None:
    rgb = np.zeros((100, 16, 3), dtype=np.uint8)
    expected_rows = np.asarray([10, 30, 50, 70, 90])
    rgb[expected_rows, :, 1] = 180
    map_x = np.tile(np.arange(16, dtype=np.float32), (100, 1))
    map_y = np.tile(np.arange(100, dtype=np.float32)[:, None], (1, 16))

    rows, source_xy, contrast, valid, _, flipped = detect_led_anchor_lattice(
        rgb, map_x, map_y, np.ones((100, 16), dtype=bool)
    )

    assert valid
    assert not flipped
    np.testing.assert_allclose(rows, expected_rows, atol=1.0)
    assert np.all(np.diff(source_xy[:, 1]) > 0.0)
    assert np.all(contrast > 7.5)


def test_support_aware_response_excludes_unmeasured_pixels() -> None:
    unloaded = np.zeros((4, 6, 3), dtype=np.uint8)
    current = np.zeros_like(unloaded)
    current[:, :3, 1] = 10
    current[:, 3:, 1] = 255
    support = np.zeros((4, 6), dtype=bool)
    support[:, :3] = True
    config = ContactLocalizationConfig(
        unloaded_frame_count=2,
        brightest_fraction=1.0,
        longitudinal_smoothing_sigma=0.0,
    )

    profile, saturation_250, saturation_255 = fig6abc._support_aware_response(
        current, unloaded, support, config
    )

    np.testing.assert_allclose(profile, 10.0)
    assert saturation_250 == 0.0
    assert saturation_255 == 0.0


def test_loaded_and_reference_use_identical_registered_coordinates() -> None:
    rgb = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    map_x = np.tile(np.arange(4, dtype=np.float32), (3, 1))
    map_y = np.tile(np.arange(3, dtype=np.float32)[:, None], (1, 4))
    registration = AnchorRegistration(
        calibration_index=0,
        session_index=0,
        capture_id="capture",
        source_mode="test",
        valid=True,
        status="valid",
        map_x=map_x,
        map_y=map_y,
        support_mask=np.ones((3, 4), dtype=bool),
        reference_rgb=rgb,
        led_rows=np.arange(5, dtype=np.float64),
        led_source_xy_px=np.zeros((5, 2), dtype=np.float64),
        tooth_contrast_dn=np.ones(5),
        source_mask_overlap=1.0,
        source_paths=("reference.png",),
    )

    loaded = fig6abc._warp_registration(rgb, registration)
    reference = fig6abc._warp_registration(registration.reference_rgb, registration)

    np.testing.assert_array_equal(loaded, reference)


def test_split_enumeration_is_deterministic_and_episode_level() -> None:
    first = enumerate_repetition_splits((5, 2, 1, 4, 3))
    second = enumerate_repetition_splits((1, 2, 3, 4, 5))

    assert first == second
    assert len(first) == 5 * (1 + 4 + 6 + 4 + 1)
    assert all(
        set(row["calibration_repetitions"]).isdisjoint(row["test_repetitions"])
        for row in first
    )


def test_optional_calibration_reuses_observations_and_holds_out_repetition(
    tmp_path: Path,
) -> None:
    observations = _complete_observations()

    predictions, _, manifest = evaluate_optional_calibration(
        observations, _config(tmp_path)
    )

    assert {row["feature_protocol"] for row in predictions} == {fig6abc.PROTOCOL_ID}
    assert {row["feature_observation_id"] for row in predictions} == {
        item.observation_id for item in observations
    }
    for split in manifest["split_records"]:
        assert set(split["training_observation_ids"]).isdisjoint(
            split["test_observation_ids"]
        )


def test_response_and_thresholded_evidence_are_distinct() -> None:
    response = np.linspace(0.0, 8.0, fig6abc.PROFILE_BIN_COUNT)
    reference = UnloadedOpticalReference(
        canonical_rgb=np.zeros((fig6abc.PROFILE_BIN_COUNT, 1, 3), dtype=np.float32),
        response_center_dn=np.zeros(fig6abc.PROFILE_BIN_COUNT),
        response_sigma_dn=np.ones(fig6abc.PROFILE_BIN_COUNT),
        response_threshold_dn=np.full(fig6abc.PROFILE_BIN_COUNT, 4.0),
        frame_count=2,
    )

    result = localize_response_profile(
        response,
        reference,
        np.linspace(0.2, 0.8, 5),
        ContactLocalizationConfig(unloaded_frame_count=2),
    )

    np.testing.assert_array_equal(result.response_profile, response)
    assert not np.array_equal(result.response_profile, result.evidence_profile)
    assert np.all(result.evidence_profile[response <= 4.0] == 0.0)
    assert result.contact_detected

    stricter_reference = replace(
        reference,
        response_threshold_dn=np.full(fig6abc.PROFILE_BIN_COUNT, 10.0),
    )
    below_gate = localize_response_profile(
        response,
        stricter_reference,
        np.linspace(0.2, 0.8, 5),
        ContactLocalizationConfig(unloaded_frame_count=2),
    )
    assert not below_gate.contact_detected


def test_calibrated_decoder_uses_response_not_thresholded_evidence(
    tmp_path: Path,
) -> None:
    observations = [
        replace(item, evidence_profile=np.zeros_like(item.evidence_profile))
        for item in _complete_observations()
    ]

    predictions, _, _ = evaluate_optional_calibration(
        observations, _config(tmp_path)
    )
    calibrated = [
        row
        for row in predictions
        if row["calibration_contacts_per_location"] == 4
    ]

    assert all(row["template_localization_valid"] for row in calibrated)
    assert all(float(row["absolute_error_mm"]) == 0.0 for row in calibrated)
    assert {
        row["calibrated_decoding_feature"] for row in calibrated
    } == {"pre-threshold registered response_profile"}


def test_evidence_threshold_change_does_not_change_calibrated_predictions(
    tmp_path: Path,
) -> None:
    observations = _complete_observations()
    altered = [
        replace(item, evidence_profile=100.0 - item.evidence_profile)
        for item in observations
    ]

    original, _, _ = evaluate_optional_calibration(observations, _config(tmp_path))
    changed, _, _ = evaluate_optional_calibration(altered, _config(tmp_path))

    original_calibrated = [
        (row["split_id"], row["observation_id"], row["predicted_hole_index"])
        for row in original
        if row["calibration_contacts_per_location"] > 0
    ]
    changed_calibrated = [
        (row["split_id"], row["observation_id"], row["predicted_hole_index"])
        for row in changed
        if row["calibration_contacts_per_location"] > 0
    ]

    assert changed_calibrated == original_calibrated


def test_contact_validity_gate_remains_independent_of_response_feature(
    tmp_path: Path,
) -> None:
    observations = [
        replace(item, contact_valid=False, contact_status="failed contact gate")
        if item.hole_index == 1 and item.repetition_index == 1
        else item
        for item in _complete_observations()
    ]

    predictions, summary, _ = evaluate_optional_calibration(
        observations, _config(tmp_path)
    )
    failed = next(
        row
        for row in predictions
        if row["calibration_contacts_per_location"] == 4
        and row["outer_repetition"] == 1
        and row["true_hole_index"] == 1
    )
    k4_summary = next(
        row for row in summary if row["calibration_contacts_per_location"] == 4
    )

    assert not failed["template_localization_valid"]
    assert failed["failure_reason"] == "failed contact gate"
    assert k4_summary["valid_contact_count"] == 29
    assert np.isclose(k4_summary["contact_coverage"], 29.0 / 30.0)


def test_diffuse_geometry_failure_does_not_block_template_decoder(
    tmp_path: Path,
) -> None:
    observations = [
        replace(
            item,
            geometry_position_mm=None,
            geometry_localization_valid=False,
            geometry_localization_status="invalid: response is spatially diffuse",
        )
        for item in _complete_observations()
    ]

    predictions, _, _ = evaluate_optional_calibration(
        observations, _config(tmp_path)
    )
    k0 = [
        row for row in predictions if row["calibration_contacts_per_location"] == 0
    ]
    k4 = [
        row for row in predictions if row["calibration_contacts_per_location"] == 4
    ]

    assert all(row["contact_valid"] for row in k0)
    assert all(not row["geometry_localization_valid"] for row in k0)
    assert all(not row["valid_prediction"] for row in k0)
    assert all(row["template_localization_valid"] for row in k4)
    assert all(row["valid_prediction"] for row in k4)


def test_k0_is_unchanged_when_registered_profiles_change(tmp_path: Path) -> None:
    observations = _complete_observations()
    altered = [
        replace(
            item,
            response_profile=np.roll(item.response_profile, 17),
            evidence_profile=np.roll(item.evidence_profile, -11),
        )
        for item in observations
    ]

    original, _, _ = evaluate_optional_calibration(observations, _config(tmp_path))
    changed, _, _ = evaluate_optional_calibration(altered, _config(tmp_path))

    original_k0 = [
        (row["observation_id"], row["predicted_position_mm"])
        for row in original
        if row["calibration_contacts_per_location"] == 0
    ]
    changed_k0 = [
        (row["observation_id"], row["predicted_position_mm"])
        for row in changed
        if row["calibration_contacts_per_location"] == 0
    ]
    assert changed_k0 == original_k0


def test_summary_reproduces_from_csv_predictions(
    tmp_path: Path,
) -> None:
    predictions, summary, _ = evaluate_optional_calibration(
        _complete_observations(), _config(tmp_path)
    )
    path = fig6abc._write_csv(tmp_path / "predictions.csv", predictions)

    reproduced = summarize_optional_calibration_predictions(
        fig6abc.read_csv(path), _config(tmp_path)
    )

    assert reproduced == summary


def test_geometry_prediction_does_not_use_contact_label(tmp_path: Path) -> None:
    original = _observation(1, 1)
    relabeled = replace(original, hole_index=6)

    original_predictions, _, _ = evaluate_optional_calibration(
        [original], _config(tmp_path)
    )
    relabeled_predictions, _, _ = evaluate_optional_calibration(
        [relabeled], _config(tmp_path)
    )

    assert original_predictions[0]["predicted_position_mm"] == 1.0
    assert relabeled_predictions[0]["predicted_position_mm"] == 1.0
    assert original_predictions[0]["true_position_mm"] == 0.0
    assert relabeled_predictions[0]["true_position_mm"] == 50.0


def test_failed_calibration_is_not_replaced_from_outside_budget(
    tmp_path: Path,
) -> None:
    observations = _complete_observations()
    observations = [
        replace(item, contact_valid=False, contact_status="failed QC")
        if item.hole_index == 1 and item.repetition_index == 2
        else item
        for item in observations
    ]

    predictions, _, manifest = evaluate_optional_calibration(
        observations, _config(tmp_path)
    )
    split = next(
        row
        for row in manifest["split_records"]
        if row["outer_repetition"] == 1
        and row["calibration_contacts_per_location"] == 1
        and row["calibration_repetitions"] == [2]
    )
    selected_predictions = [
        row for row in predictions if row["split_id"] == split["split_id"]
    ]

    assert "invalid calibration episode" in split["template_failure"]
    assert split["calibration_repetitions"] == [2]
    assert all(not row["valid_prediction"] for row in selected_predictions)
    assert all("invalid calibration episode" in row["failure_reason"] for row in selected_predictions)


def test_invalid_k0_estimate_and_conditional_mae_denominators_are_explicit(
    tmp_path: Path,
) -> None:
    observations = _complete_observations()
    observations = [
        replace(
            item,
            geometry_valid=False,
            geometry_position_mm=None,
            geometry_localization_valid=False,
            geometry_localization_status="failed geometry QC",
        )
        if item.hole_index == 1 and item.repetition_index == 1
        else item
        for item in observations
    ]

    predictions, summary, _ = evaluate_optional_calibration(
        observations, _config(tmp_path)
    )
    k0_predictions = [
        row for row in predictions if row["calibration_contacts_per_location"] == 0
    ]
    k0_summary = next(
        row for row in summary if row["calibration_contacts_per_location"] == 0
    )
    errors = [
        float(row["absolute_error_mm"])
        for row in k0_predictions
        if row["valid_prediction"]
    ]

    assert len(k0_predictions) == 30
    assert len(errors) == 29
    assert k0_summary["attempted_test_contacts"] == 30
    assert k0_summary["geometry_valid_count"] == 29
    assert k0_summary["template_valid_count"] == ""
    assert k0_summary["valid_prediction_count"] == 29
    assert np.isclose(k0_summary["coverage_mean"], 29.0 / 30.0)
    assert np.isclose(k0_summary["localization_mae_mean_mm"], np.mean(errors))
    failed = next(row for row in k0_predictions if not row["valid_prediction"])
    assert failed["absolute_error_mm"] == ""
    assert failed["failure_reason"] == "failed geometry QC"


def test_revised_observation_artifact_uses_5n_only(tmp_path: Path) -> None:
    observation = _observation(1, 1)
    path = tmp_path / "observations.npz"

    fig6abc.write_observations_npz(
        path,
        [observation],
        {("material", "baseline"): np.linspace(-5.0, 55.0, fig6abc.PROFILE_BIN_COUNT)},
    )

    with np.load(path, allow_pickle=False) as data:
        assert float(data["target_force_n"]) == 5.0
        assert "low_force_n" not in data.files
        assert "response_profiles" in data.files
        assert "evidence_profiles" in data.files
        assert "contact_valid" in data.files
        assert "geometry_localization_valid" in data.files
        assert str(data["feature_protocol"]) == fig6abc.PROTOCOL_ID
        assert "2 N" not in str(data["feature_protocol"])


def test_cache_invalidates_when_relevant_config_changes(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: 1\n", encoding="utf-8")
    artifact_path = tmp_path / "dataset.h5"
    artifact_path.write_bytes(b"placeholder")
    output = tmp_path / fig6abc.OUTPUT_SUBDIRECTORY
    output.mkdir()
    for name in (
        "fig6a_contact_state_variability.csv",
        "fig6b_distinguishability_values.csv",
        "fig6c_contact_observations.npz",
        "fig6c_geometry_anchors.csv",
        "fig6c_split_manifest.json",
        "fig6c_predictions.csv",
        "fig6c_summary.csv",
    ):
        (output / name).write_bytes(b"present")
    monkeypatch.setitem(fig6abc.HISTORY_ROOTS, "material", tmp_path)

    monkeypatch.setattr(
        fig6abc,
        "_relevant_raw_input_digest",
        lambda _path: ("raw-digest", 1),
    )

    def fingerprint(
        current_config: Path,
        _artifact: Path,
        _inputs,
        raw_digest: str,
    ) -> tuple[str, dict[str, str]]:
        payload = current_config.read_bytes() + raw_digest.encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), {}

    monkeypatch.setattr(fig6abc, "_protocol_fingerprint", fingerprint)
    initial, _ = fingerprint(config_path, artifact_path, (), "raw-digest")
    (output / "fig6c_provenance.json").write_text(
        json.dumps({"protocol_fingerprint": initial}), encoding="utf-8"
    )

    assert fig6abc.cached_artifacts_current(
        config, config_path, artifact_path=artifact_path
    )
    config_path.write_text("version: 2\n", encoding="utf-8")
    assert not fig6abc.cached_artifacts_current(
        config, config_path, artifact_path=artifact_path
    )

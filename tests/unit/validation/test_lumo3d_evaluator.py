from __future__ import annotations

from pathlib import Path

from validation.optimization.lumo3d_evaluator import (
    CONTACT_STATE_SEPARATION_OBJECTIVE_NAME,
    LUMO3D_EVALUATION_CONTRACT,
    LUMO3D_EVALUATION_CONTRACT_ID,
    LUMO3D_OBSERVATION_LEVEL,
    Lumo3DEvaluation,
    create_lumo3d_study,
)


def test_lumo3d_contract_is_explicitly_three_state_and_maximize_oriented() -> None:
    assert LUMO3D_EVALUATION_CONTRACT["contact"]["normalized_locations"] == (
        0.25,
        0.5,
        0.75,
    )
    assert LUMO3D_EVALUATION_CONTRACT["objective"]["name"] == (
        CONTACT_STATE_SEPARATION_OBJECTIVE_NAME
    )
    assert LUMO3D_EVALUATION_CONTRACT["objective"]["direction"] == "maximize"
    assert LUMO3D_OBSERVATION_LEVEL.startswith("FULL_3D")
    assert LUMO3D_EVALUATION_CONTRACT_ID.startswith("lumo3d-multi-contact-v1-")


def test_lumo3d_study_reuses_all_five_active_production_variables(tmp_path: Path) -> None:
    study = create_lumo3d_study(tmp_path)
    assert [variable.name for variable in study.design_space.active_variables] == [
        "flat_pad_height",
        "stem_width",
        "stem_height",
        "void_width",
        "void_height",
    ]
    assert study.create_evaluator().objective_name == CONTACT_STATE_SEPARATION_OBJECTIVE_NAME


def test_lumo3d_result_keeps_pairwise_objective_separate_from_minimum_auc() -> None:
    result = Lumo3DEvaluation(
        status="success",
        objective_value=0.12,
        pairwise_distance_matrix=((0.0, 0.12, 0.2), (0.12, 0.0, 0.15), (0.2, 0.15, 0.0)),
        contact_states=(),
        mechanics_diagnostics=(),
        optical_diagnostics=(),
        diagnostics={"objective_name": CONTACT_STATE_SEPARATION_OBJECTIVE_NAME},
    )
    assert result.score == result.objective_value == 0.12
    assert not hasattr(result, "minimum_auc")

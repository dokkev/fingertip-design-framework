"""Status and setup contracts for the bounded validation BO report."""

from types import SimpleNamespace

from lumo.mechanics_contract import MechanicsContract
from validation.optimization.lumo6d_test_bo import _search_mechanics, _status_contract


def test_status_contract_preserves_current_evaluation_taxonomy() -> None:
    assert _status_contract("success") == "valid_success"
    assert _status_contract("invalid_design") == "geometry_rejected"
    assert _status_contract("domain_incompatible") == "domain_incompatible"
    assert _status_contract("mesh_failure") == "geometry_rejected"
    assert _status_contract("mechanics_failure") == "mechanics_failed"
    assert _status_contract("optics_failure") == "optics_failed"
    assert _status_contract("duplicate_skipped") == "duplicate_skipped"


def test_removed_failure_vocabulary_is_not_silently_current() -> None:
    assert _status_contract("fea_failure") == "infrastructure_failed"


def test_test_bo_serializes_the_mechanics_contract_at_the_setup_boundary() -> None:
    contract = MechanicsContract()
    study = SimpleNamespace(
        create_evaluator=lambda: SimpleNamespace(mechanics_contract=contract)
    )

    search_mechanics = _search_mechanics(study)

    assert search_mechanics == contract.to_dict()

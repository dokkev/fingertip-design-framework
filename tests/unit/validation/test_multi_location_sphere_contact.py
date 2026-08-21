from dataclasses import replace

import pytest

from lumo.contact import DEFAULT_FIRST_CONTACT_SETTINGS
from lumo.finger import FingertipParameters, ViscoelasticParameters
from lumo.mechanics_contract import MechanicsContract
from lumo.physics.contracts import VBDDeterminismMode
from validation.physics.multi_location_sphere_contact import (
    _direct_execution_settings,
)


def test_direct_replay_translates_complete_mechanics_contract() -> None:
    parameters = FingertipParameters(
        viscoelastic=ViscoelasticParameters(
            density_kg_m3=1100.0,
            k_mu_pa=120000.0,
            k_lambda_pa=130000.0,
            k_damp=12.0,
        )
    )
    first_contact = replace(DEFAULT_FIRST_CONTACT_SETTINGS, coarse_step_mm=0.2)
    contract = MechanicsContract(
        sphere_subdivisions=2,
        max_load_increment_mm=0.125,
        vbd_iterations=17,
        deterministic_mode=VBDDeterminismMode.RUN_TO_RUN,
        dt_s=0.002,
        soft_contact_margin_mm=0.03,
        soft_contact_ke=900.0,
        soft_contact_kd=8.0,
        soft_contact_mu=0.2,
        rigid_sdf_target_voxel_mm=0.2,
        first_contact=first_contact,
    )

    resolved_contact, newton, indentation = _direct_execution_settings(
        contract,
        parameters,
        device="cuda:1",
        travel_mm=0.5,
        support_vertex_indices=(1, 3),
    )

    assert resolved_contact is first_contact
    assert newton.device == "cuda:1"
    assert newton.iterations == 17
    assert newton.dt == pytest.approx(0.002)
    assert newton.deterministic_mode is VBDDeterminismMode.RUN_TO_RUN
    assert newton.density == pytest.approx(1100.0)
    assert newton.k_mu == pytest.approx(120000.0)
    assert newton.k_lambda == pytest.approx(130000.0)
    assert newton.k_damp == pytest.approx(12.0)
    assert newton.fixed_vertex_indices == (1, 3)
    assert indentation.load_steps == 4
    assert indentation.soft_contact_margin_mm == pytest.approx(0.03)
    assert indentation.soft_contact_ke == pytest.approx(900.0)
    assert indentation.soft_contact_kd == pytest.approx(8.0)
    assert indentation.soft_contact_mu == pytest.approx(0.2)
    assert indentation.rigid_sdf_target_voxel_mm == pytest.approx(0.2)

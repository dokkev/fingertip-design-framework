"""Central Phase 4M Kratos solver and contact settings."""

from __future__ import annotations

import json
from typing import Any

MIXED_PAD_ELEMENT = "TotalLagrangianMixedVolumetricStrainElement2D3N"
CARRIER_ELEMENT = "TotalLagrangianElement2D3N"
CONSTITUTIVE_LAW = "HyperElasticPlaneStrain2DLaw"
MORTAR_TYPE = "ALMContactFrictionless"

YOUNG_MODULUS_MPA = 1.0
POISSON_RATIO = 0.49
THICKNESS_MM = 1.0
RELATIVE_TOLERANCE = 1.0e-6
ABSOLUTE_TOLERANCE = 1.0e-9
MAXIMUM_NEWTON_ITERATIONS = 35

CONTACT_GROUPS = (
    ("PadCutoutLeft", "StemLeft"),
    ("PadCutoutRight", "StemRight"),
    ("PadCutoutBottom", "StemBottom"),
)


def build_project_parameters_data() -> dict[str, Any]:
    """Build the single source for the Phase 4M initialization configuration."""
    assume_master_slave = {
        str(index): [slave]
        for index, (slave, _) in enumerate(CONTACT_GROUPS)
    }
    contact_model_part = {
        str(index): [slave, master]
        for index, (slave, master) in enumerate(CONTACT_GROUPS)
    }
    return {
        "problem_data": {
            "problem_name": "phase4m_fingertip_initialization",
            "parallel_type": "OpenMP",
            "start_time": 0.0,
            "end_time": 1.0,
            "echo_level": 0,
        },
        "solver_settings": {
            "model_part_name": "Structure",
            "domain_size": 2,
            "solver_type": "Static",
            "echo_level": 0,
            "analysis_type": "non_linear",
            "model_import_settings": {"input_type": "use_input_model_part"},
            "material_import_settings": {"materials_filename": ""},
            "time_stepping": {"time_step": 1.0},
            "volumetric_strain_dofs": True,
            "contact_settings": {
                "mortar_type": MORTAR_TYPE,
                "ensure_contact": False,
                "silent_strategy": True,
                "simplified_semi_smooth_newton": False,
                "fancy_convergence_criterion": False,
                "print_convergence_criterion": False,
            },
            "clear_storage": True,
            "reform_dofs_at_each_step": True,
            "compute_reactions": True,
            "move_mesh_flag": True,
            "convergence_criterion": "contact_residual_criterion",
            "displacement_relative_tolerance": RELATIVE_TOLERANCE,
            "displacement_absolute_tolerance": ABSOLUTE_TOLERANCE,
            "residual_relative_tolerance": RELATIVE_TOLERANCE,
            "residual_absolute_tolerance": ABSOLUTE_TOLERANCE,
            "max_iteration": MAXIMUM_NEWTON_ITERATIONS,
            "builder_and_solver_settings": {
                "type": "block",
                "advanced_settings": {},
            },
            "solving_strategy_settings": {
                "type": "newton_raphson",
                "advanced_settings": {},
            },
            "linear_solver_settings": {
                "solver_type": "skyline_lu_factorization"
            },
        },
        "processes": {
            "contact_process_list": [
                {
                    "python_module": "alm_contact_process",
                    "kratos_module": (
                        "KratosMultiphysics.ContactStructuralMechanicsApplication"
                    ),
                    "process_name": "ALMContactProcess",
                    "Parameters": {
                        "model_part_name": "Structure",
                        "assume_master_slave": assume_master_slave,
                        "contact_model_part": contact_model_part,
                        "contact_type": "Frictionless",
                    },
                }
            ]
        },
    }


def build_project_parameters_json() -> str:
    """Serialize the shared settings for ``KM.Parameters``."""
    return json.dumps(build_project_parameters_data())

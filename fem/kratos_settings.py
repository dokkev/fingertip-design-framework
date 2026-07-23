"""Central Phase 4M Kratos solver and contact settings."""

from __future__ import annotations

import json
from typing import Any, Literal

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

# The order is a runtime contract: Kratos creates ContactSubN and
# ComputingContactSubN with the same numeric key.  Post-processing uses these
# model parts directly instead of classifying generated conditions by position.
INDENTATION_CONTACT_GROUPS = (
    ("external_pad_indenter", "PadOuterArc", "IndenterContactArc"),
    ("internal_left", "PadCutoutLeft", "StemLeft"),
    ("internal_right", "PadCutoutRight", "StemRight"),
    ("internal_bottom", "PadCutoutBottom", "StemBottom"),
)

InternalContactConfiguration = Literal[
    "none",
    "bottom_only",
    "sides_separate",
    "three_pairs",
    "continuous_u",
    "left_only",
    "right_only",
]

INTERNAL_CONTACT_CONFIGURATIONS = (
    "none",
    "bottom_only",
    "sides_separate",
    "three_pairs",
    "continuous_u",
)

_EXTERNAL_CONTACT_GROUP = (
    "external_pad_indenter",
    "PadOuterArc",
    "IndenterContactArc",
)

_INTERNAL_CONTACT_GROUPS = {
    "none": (),
    "bottom_only": (
        ("internal_bottom", "PadCutoutBottom", "StemBottom"),
    ),
    "sides_separate": (
        ("internal_left", "PadCutoutLeft", "StemLeft"),
        ("internal_right", "PadCutoutRight", "StemRight"),
    ),
    "three_pairs": (
        ("internal_left", "PadCutoutLeft", "StemLeft"),
        ("internal_right", "PadCutoutRight", "StemRight"),
        ("internal_bottom", "PadCutoutBottom", "StemBottom"),
    ),
    "continuous_u": (
        ("internal_u", "PadInternalU", "StemInternalU"),
    ),
    # Diagnostic-only follow-ups, used only when the two-side case fails.
    "left_only": (
        ("internal_left", "PadCutoutLeft", "StemLeft"),
    ),
    "right_only": (
        ("internal_right", "PadCutoutRight", "StemRight"),
    ),
}


def validate_internal_contact_configuration(
    configuration: str,
) -> InternalContactConfiguration:
    """Validate and return one supported internal-contact configuration."""
    if configuration not in _INTERNAL_CONTACT_GROUPS:
        supported = ", ".join(sorted(_INTERNAL_CONTACT_GROUPS))
        raise ValueError(
            f"unsupported internal contact configuration {configuration!r}; "
            f"expected one of: {supported}"
        )
    return configuration  # type: ignore[return-value]


def indentation_contact_groups(
    configuration: str = "three_pairs",
) -> tuple[tuple[str, str, str], ...]:
    """Return the indexed external and selected internal ALM surface pairs."""
    validated = validate_internal_contact_configuration(configuration)
    return (_EXTERNAL_CONTACT_GROUP, *_INTERNAL_CONTACT_GROUPS[validated])


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


def build_indentation_project_parameters_data(
    number_of_steps: int,
    internal_contact_configuration: str = "three_pairs",
) -> dict[str, Any]:
    """Build the common Phase 4I nonlinear solve and four-pair ALM settings."""
    if (
        not isinstance(number_of_steps, int)
        or isinstance(number_of_steps, bool)
        or number_of_steps <= 0
    ):
        raise ValueError("number_of_steps must be a positive integer")
    data = build_project_parameters_data()
    data["problem_data"].update(
        {
            "problem_name": "phase4i_central_indentation",
            "end_time": float(number_of_steps),
        }
    )
    data["solver_settings"]["time_stepping"] = {"time_step": 1.0}
    # Preserve the direct solver used by the validated Phase 3R/4M stack.
    data["solver_settings"]["linear_solver_settings"] = {
        "solver_type": "skyline_lu_factorization"
    }
    groups = indentation_contact_groups(internal_contact_configuration)
    contact_process = data["processes"]["contact_process_list"][0]["Parameters"]
    contact_process["assume_master_slave"] = {
        str(index): [slave]
        for index, (_, slave, _) in enumerate(groups)
    }
    contact_process["contact_model_part"] = {
        str(index): [slave, master]
        for index, (_, slave, master) in enumerate(groups)
    }
    return data


def build_indentation_project_parameters_json(
    number_of_steps: int,
    internal_contact_configuration: str = "three_pairs",
) -> str:
    """Serialize the common Phase 4I settings for ``KM.Parameters``."""
    return json.dumps(
        build_indentation_project_parameters_data(
            number_of_steps, internal_contact_configuration
        )
    )

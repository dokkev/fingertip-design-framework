"""Central Phase 4M Kratos solver and contact settings."""

from __future__ import annotations

import json
from dataclasses import dataclass
import math
from typing import Any, Literal

MIXED_PAD_ELEMENT = "TotalLagrangianMixedVolumetricStrainElement2D3N"
CARRIER_ELEMENT = "TotalLagrangianElement2D3N"
CONSTITUTIVE_LAW = "HyperElasticPlaneStrain2DLaw"
MORTAR_TYPE = "ALMContactFrictionless"

# Backend defaults retained for the constrained rigid carrier/indenter and
# standalone validation benchmarks. Production compliant-pad properties come
# from FingertipParameters through the mesh.
YOUNG_MODULUS_MPA = 1.0
POISSON_RATIO = 0.49
THICKNESS_MM = 1.0
RELATIVE_TOLERANCE = 1.0e-6
ABSOLUTE_TOLERANCE = 1.0e-9
MAXIMUM_NEWTON_ITERATIONS = 35


@dataclass(frozen=True)
class IndentationSolverSettings:
    """Explicit Phase 4I solver knobs used by benchmark A/B studies.

    The default instance preserves the trusted production configuration.  The
    settings are passed only to a requested indentation run; importing this
    module or calling the existing APIs does not change the defaults.
    """

    relative_tolerance: float = RELATIVE_TOLERANCE
    absolute_tolerance: float = ABSOLUTE_TOLERANCE
    maximum_newton_iterations: int = MAXIMUM_NEWTON_ITERATIONS
    linear_solver_type: str = "skyline_lu_factorization"
    reform_dofs_at_each_step: bool = True
    compute_reactions: bool = True
    clear_storage: bool = True

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.relative_tolerance)
            or not math.isfinite(self.absolute_tolerance)
            or self.relative_tolerance <= 0.0
            or self.absolute_tolerance <= 0.0
        ):
            raise ValueError("solver tolerances must be positive")
        if (
            not isinstance(self.maximum_newton_iterations, int)
            or isinstance(self.maximum_newton_iterations, bool)
            or self.maximum_newton_iterations <= 0
        ):
            raise ValueError("maximum_newton_iterations must be positive")
        if not self.linear_solver_type:
            raise ValueError("linear_solver_type must be non-empty")


DEFAULT_INDENTATION_SOLVER_SETTINGS = IndentationSolverSettings()

# The order is a runtime contract: Kratos creates ContactSubN and
# ComputingContactSubN with the same numeric key.  Post-processing uses these
# model parts directly instead of classifying generated conditions by position.
# Complete semantic contact catalog used by mesh/runtime diagnostics. Production
# selection is controlled by ``indentation_contact_groups`` below.
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

BasalInterface = Literal["bonded", "explicit_contact", "free"]

BASAL_INTERFACES = ("bonded", "explicit_contact", "free")

INTERNAL_CONTACT_CONFIGURATIONS = (
    "none",
    "bottom_only",
    "sides_separate",
    "three_pairs",
    "continuous_u",
    "left_only",
    "right_only",
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


def validate_basal_interface(interface: str) -> BasalInterface:
    """Validate the physical basal-interface condition independently."""
    if interface not in BASAL_INTERFACES:
        supported = ", ".join(BASAL_INTERFACES)
        raise ValueError(
            f"unsupported basal interface {interface!r}; "
            f"expected one of: {supported}"
        )
    return interface  # type: ignore[return-value]


def validate_basal_interface_configuration(
    basal_interface: str,
    internal_contact_configuration: str,
) -> tuple[BasalInterface, InternalContactConfiguration]:
    """Validate the independent basal and internal-contact selections."""
    basal = validate_basal_interface(basal_interface)
    internal = validate_internal_contact_configuration(
        internal_contact_configuration
    )
    has_bottom_contact = any(
        name == "internal_bottom"
        for name, _, _ in indentation_contact_groups(internal)
    )
    if basal in ("bonded", "free") and has_bottom_contact:
        raise ValueError(
            f"basal_interface={basal!r} cannot be combined with "
            f"internal_contact={internal!r}, which registers internal_bottom"
        )
    if basal == "explicit_contact" and not has_bottom_contact:
        raise ValueError(
            "basal_interface='explicit_contact' requires an internal contact "
            "configuration containing internal_bottom"
        )
    return basal, internal


def indentation_contact_groups(
    configuration: str = "sides_separate",
) -> tuple[tuple[str, str, str], ...]:
    """Return the indexed external and selected internal ALM surface pairs."""
    validated = validate_internal_contact_configuration(configuration)
    return (_EXTERNAL_CONTACT_GROUP, *_INTERNAL_CONTACT_GROUPS[validated])


def build_project_parameters_data(
    internal_contact_configuration: str = "sides_separate",
) -> dict[str, Any]:
    """Build the Phase 4M configuration for the selected contact policy."""
    groups = indentation_contact_groups(internal_contact_configuration)
    assume_master_slave = {
        str(index): [slave] for index, (_, slave, _) in enumerate(groups)
    }
    contact_model_part = {
        str(index): [slave, master]
        for index, (_, slave, master) in enumerate(groups)
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


def build_project_parameters_json(
    internal_contact_configuration: str = "sides_separate",
) -> str:
    """Serialize the shared settings for ``KM.Parameters``."""
    return json.dumps(
        build_project_parameters_data(internal_contact_configuration)
    )


def build_indentation_project_parameters_data(
    number_of_steps: int,
    internal_contact_configuration: str = "sides_separate",
    solver_settings: IndentationSolverSettings | None = None,
) -> dict[str, Any]:
    """Build the common Phase 4I nonlinear solve and selected ALM settings."""
    if (
        not isinstance(number_of_steps, int)
        or isinstance(number_of_steps, bool)
        or number_of_steps <= 0
    ):
        raise ValueError("number_of_steps must be a positive integer")
    data = build_project_parameters_data()
    selected_solver_settings = (
        DEFAULT_INDENTATION_SOLVER_SETTINGS
        if solver_settings is None
        else solver_settings
    )
    if not isinstance(selected_solver_settings, IndentationSolverSettings):
        raise TypeError("solver_settings must be IndentationSolverSettings or None")
    data["problem_data"].update(
        {
            "problem_name": "phase4i_central_indentation",
            "end_time": float(number_of_steps),
        }
    )
    data["solver_settings"]["time_stepping"] = {"time_step": 1.0}
    # Preserve the direct solver used by the validated Phase 3R/4M stack.
    data["solver_settings"]["linear_solver_settings"] = {
        "solver_type": selected_solver_settings.linear_solver_type
    }
    data["solver_settings"].update(
        {
            "clear_storage": selected_solver_settings.clear_storage,
            "reform_dofs_at_each_step": selected_solver_settings.reform_dofs_at_each_step,
            "compute_reactions": selected_solver_settings.compute_reactions,
            "displacement_relative_tolerance": selected_solver_settings.relative_tolerance,
            "displacement_absolute_tolerance": selected_solver_settings.absolute_tolerance,
            "residual_relative_tolerance": selected_solver_settings.relative_tolerance,
            "residual_absolute_tolerance": selected_solver_settings.absolute_tolerance,
            "max_iteration": selected_solver_settings.maximum_newton_iterations,
        }
    )
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
    internal_contact_configuration: str = "sides_separate",
    solver_settings: IndentationSolverSettings | None = None,
) -> str:
    """Serialize the common Phase 4I settings for ``KM.Parameters``."""
    return json.dumps(
        build_indentation_project_parameters_data(
            number_of_steps, internal_contact_configuration, solver_settings
        )
    )

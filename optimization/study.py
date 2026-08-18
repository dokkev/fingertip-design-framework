"""Immutable fixed scientific configuration for one optimization study."""

from __future__ import annotations

from dataclasses import dataclass

from fem.kratos_settings import validate_basal_interface_configuration
from mesh import MeshSettings, mesh_settings_for_level
from model import Fingertip, FingertipParameters, LED, OpticalMaterial
from optics import IndenterOptics
from optics.transport3d import Transport3DSettings

from optimization.design_space import (
    OPTIMIZABLE_PARAMETER_NAMES,
    DesignSpace,
    DesignVariable,
)
from optimization.evaluator import DesignEvaluator
from optimization.scenarios import ScenarioGrid


PRODUCTION_FIXED_FLAT_PAD_WIDTH_MM = 30.0
PRODUCTION_SEARCH_BOUNDS: tuple[tuple[str, float, float], ...] = (
    ("flat_pad_height", 3.5, 6.5),
    ("stem_width", 6.5, 9.0),
    ("stem_height", 5.0, 7.5),
    ("void_width", 0.5, 2.0),
)


def _production_design_space(
    nominal_parameters: FingertipParameters | None = None,
) -> DesignSpace:
    nominal = FingertipParameters() if nominal_parameters is None else nominal_parameters
    bounds = {name: (lower, upper) for name, lower, upper in PRODUCTION_SEARCH_BOUNDS}
    variables = tuple(
        DesignVariable(
            name,
            True,
            bounds[name][0],
            bounds[name][1],
        )
        for name in OPTIMIZABLE_PARAMETER_NAMES
    )
    return DesignSpace(nominal, variables)


@dataclass(frozen=True)
class OptimizationStudy:
    """One complete fixed scientific experiment, independent of an optimizer."""

    design_space: DesignSpace
    scenario_grid: ScenarioGrid
    mesh_settings: MeshSettings
    trace_settings: Transport3DSettings
    led: LED
    optical: OpticalMaterial
    indenter_optics: IndenterOptics
    fem_steps: int = 48
    internal_contact: str = "sides_separate"
    basal_interface: str = "bonded"

    def __post_init__(self) -> None:
        if not isinstance(self.design_space, DesignSpace):
            raise TypeError("design_space must be a DesignSpace")
        if not isinstance(self.scenario_grid, ScenarioGrid):
            raise TypeError("scenario_grid must be a ScenarioGrid")
        if not isinstance(self.mesh_settings, MeshSettings):
            raise TypeError("mesh_settings must be MeshSettings")
        if not isinstance(self.trace_settings, Transport3DSettings):
            raise TypeError("trace_settings must be Transport3DSettings")
        if not isinstance(self.led, LED):
            raise TypeError("led must be an LED")
        if not isinstance(self.optical, OpticalMaterial):
            raise TypeError("optical must be an OpticalMaterial")
        if self.indenter_optics is None:
            raise ValueError(
                "production OptimizationStudy requires explicit indenter_optics"
            )
        if not isinstance(self.indenter_optics, IndenterOptics):
            raise TypeError("indenter_optics must be an IndenterOptics")
        if (
            not isinstance(self.fem_steps, int)
            or isinstance(self.fem_steps, bool)
            or self.fem_steps != 48
        ):
            raise ValueError("production OptimizationStudy requires fem_steps=48")
        if not isinstance(self.internal_contact, str) or not self.internal_contact:
            raise ValueError("internal_contact must be a non-empty string")
        if not isinstance(self.basal_interface, str) or not self.basal_interface:
            raise ValueError("basal_interface must be a non-empty string")
        basal, internal = validate_basal_interface_configuration(
            self.basal_interface,
            self.internal_contact,
        )
        object.__setattr__(self, "basal_interface", basal)
        object.__setattr__(self, "internal_contact", internal)
        if not self.design_space.active_variables:
            raise ValueError("OptimizationStudy requires at least one active variable")

        for variable in self.design_space.active_variables:
            nominal_value = getattr(
                self.design_space.nominal_parameters,
                variable.name,
            )
            if variable.lower >= variable.upper:
                raise ValueError(
                    f"{variable.name} is active but has zero search width: "
                    f"lower={variable.lower:g}, upper={variable.upper:g}"
                )
            if not variable.lower <= nominal_value <= variable.upper:
                raise ValueError(
                    f"nominal {variable.name}={nominal_value:g} is outside "
                    f"[{variable.lower:g}, {variable.upper:g}]"
                )

        # The public physical root validates the complete nominal design, including
        # fixed LED fit, without meshing, FEM, or optical transport.
        Fingertip(
            self.design_space.nominal_parameters,
            led=self.led,
            optical=self.optical,
        )

    def create_evaluator(self) -> DesignEvaluator:
        """Create a fresh evaluator bound to this study's fixed configuration."""
        return DesignEvaluator(
            self.scenario_grid,
            mesh_settings=self.mesh_settings,
            trace_settings=self.trace_settings,
            led=self.led,
            optical=self.optical,
            indenter_optics=self.indenter_optics,
            fem_steps=self.fem_steps,
            internal_contact=self.internal_contact,
            basal_interface=self.basal_interface,
        )


def create_production_study(
    *,
    nominal_parameters: FingertipParameters | None = None,
) -> OptimizationStudy:
    """Build the frozen four-variable production optimization configuration."""
    return OptimizationStudy(
        design_space=_production_design_space(nominal_parameters),
        scenario_grid=ScenarioGrid(),
        mesh_settings=mesh_settings_for_level("medium"),
        trace_settings=Transport3DSettings(mode="planar"),
        led=LED(),
        optical=OpticalMaterial(),
        indenter_optics=IndenterOptics("absorber"),
        fem_steps=48,
        internal_contact="sides_separate",
        basal_interface="bonded",
    )


__all__ = [
    "OptimizationStudy",
    "PRODUCTION_FIXED_FLAT_PAD_WIDTH_MM",
    "PRODUCTION_SEARCH_BOUNDS",
    "create_production_study",
]

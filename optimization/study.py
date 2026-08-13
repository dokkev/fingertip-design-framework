"""Immutable fixed scientific configuration for one optimization study."""

from __future__ import annotations

from dataclasses import dataclass

from mesh import MeshSettings
from model import Fingertip, LED, OpticalMaterial
from optics import TraceSettings

from optimization.design_space import DesignSpace
from optimization.evaluator import DesignEvaluator
from optimization.scenarios import ScenarioGrid


@dataclass(frozen=True)
class OptimizationStudy:
    """One complete fixed scientific experiment, independent of an optimizer."""

    design_space: DesignSpace
    scenario_grid: ScenarioGrid
    mesh_settings: MeshSettings
    trace_settings: TraceSettings
    led: LED
    optical: OpticalMaterial
    fem_steps: int = 48
    internal_contact: str = "three_pairs"

    def __post_init__(self) -> None:
        if not isinstance(self.design_space, DesignSpace):
            raise TypeError("design_space must be a DesignSpace")
        if not isinstance(self.scenario_grid, ScenarioGrid):
            raise TypeError("scenario_grid must be a ScenarioGrid")
        if not self.scenario_grid.adjacent_pairs:
            raise ValueError(
                "scenario_grid must contain at least one required adjacent pair"
            )
        if not isinstance(self.mesh_settings, MeshSettings):
            raise TypeError("mesh_settings must be MeshSettings")
        if not isinstance(self.trace_settings, TraceSettings):
            raise TypeError("trace_settings must be TraceSettings")
        if not isinstance(self.led, LED):
            raise TypeError("led must be an LED")
        if not isinstance(self.optical, OpticalMaterial):
            raise TypeError("optical must be an OpticalMaterial")
        if (
            not isinstance(self.fem_steps, int)
            or isinstance(self.fem_steps, bool)
            or self.fem_steps <= 0
        ):
            raise ValueError("fem_steps must be a positive integer")
        if not isinstance(self.internal_contact, str) or not self.internal_contact:
            raise ValueError("internal_contact must be a non-empty string")
        if not self.design_space.active_variables:
            raise ValueError("OptimizationStudy requires at least one active variable")

        for variable in self.design_space.active_variables:
            baseline_value = getattr(self.design_space.baseline, variable.name)
            if variable.lower >= variable.upper:
                raise ValueError(
                    f"{variable.name} is active but has zero search width: "
                    f"lower={variable.lower:g}, upper={variable.upper:g}"
                )
            if not variable.lower <= baseline_value <= variable.upper:
                raise ValueError(
                    f"baseline {variable.name}={baseline_value:g} is outside "
                    f"[{variable.lower:g}, {variable.upper:g}]"
                )

        # The public physical root validates the complete baseline, including
        # fixed LED fit, without meshing, FEM, or optical transport.
        Fingertip(
            self.design_space.baseline,
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
            fem_steps=self.fem_steps,
            internal_contact=self.internal_contact,
        )


__all__ = ["OptimizationStudy"]

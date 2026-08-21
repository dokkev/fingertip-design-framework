"""Public morphology-search configuration and evaluation API.

The optional Ax dependency remains below ``lumo.optimization.adapters.ax`` and
is imported only at the execution boundary. Persistence records and objective
implementation details stay in their owning modules rather than this public
surface.
"""
from lumo.optimization.design_space import (
    DesignSpace,
    DesignSpaceFeasibilityError,
    DesignVariable,
    FEASIBLE_PARAMETERIZATION_VERSION,
    LatentVariable,
    LinearConstraint,
    OptimizableParameterName,
    ParameterSpec,
)
from lumo.optimization.evaluator import Lumo3DTrajectoryEvaluator
from lumo.optimization.objectives import (
    TrajectoryObjectiveConfig,
)
from lumo.optimization.protocol import (
    DEFAULT_TRAJECTORY_PROTOCOL,
    TrajectoryEvaluationProtocol,
)

__all__ = [
    "DEFAULT_TRAJECTORY_PROTOCOL",
    "DesignSpace",
    "DesignSpaceFeasibilityError",
    "DesignVariable",
    "FEASIBLE_PARAMETERIZATION_VERSION",
    "LinearConstraint",
    "LatentVariable",
    "Lumo3DTrajectoryEvaluator",
    "OptimizableParameterName",
    "ParameterSpec",
    "TrajectoryEvaluationProtocol",
    "TrajectoryObjectiveConfig",
]

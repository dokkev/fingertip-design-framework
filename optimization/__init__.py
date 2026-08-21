"""Current morphology-search contracts.

The optional Ax dependency is imported only by ``optimization.adapters.ax`` at
the execution boundary; importing this package stays lightweight.
"""
from optimization.design_space import (
    DesignSpace,
    DesignVariable,
    LinearConstraint,
    OPTIMIZABLE_PARAMETER_NAMES,
    OptimizableParameterName,
    ParameterSpec,
    PRODUCTION_MAX_TOTAL_PAD_DEPTH_MM,
    PRODUCTION_NOMINAL_VOID_HEIGHT_MM,
    PRODUCTION_SEARCH_BOUNDS,
    PRODUCTION_LINEAR_CONSTRAINTS,
)
from optimization.evaluation_registry import (
    EvaluationRegistry,
    EvaluationRegistryRecord,
)
from optimization.objectives import (
    ObjectiveIdentifier,
    TRAJECTORY_SEPARATION_OBJECTIVE,
    TrajectoryObjectiveConfig,
    TrajectoryObservationKey,
    TrajectoryObservation,
    TrajectoryPairDistance,
    TrajectoryObjectiveResult,
    TrajectorySeparationObjective,
    compute_trajectory_objective,
    normalized_field_distance,
)
from optimization.protocol import (
    DEFAULT_TRAJECTORY_PROTOCOL,
    PROTOCOL_SCHEMA,
    TrajectoryEvaluationProtocol,
)

__all__ = [
    "DEFAULT_TRAJECTORY_PROTOCOL",
    "DesignSpace",
    "DesignVariable",
    "LinearConstraint",
    "EvaluationRegistry",
    "EvaluationRegistryRecord",
    "ObjectiveIdentifier",
    "TRAJECTORY_SEPARATION_OBJECTIVE",
    "OPTIMIZABLE_PARAMETER_NAMES",
    "OptimizableParameterName",
    "ParameterSpec",
    "PRODUCTION_MAX_TOTAL_PAD_DEPTH_MM",
    "PRODUCTION_NOMINAL_VOID_HEIGHT_MM",
    "PRODUCTION_SEARCH_BOUNDS",
    "PRODUCTION_LINEAR_CONSTRAINTS",
    "PROTOCOL_SCHEMA",
    "TrajectoryEvaluationProtocol",
    "TrajectoryObjectiveConfig",
    "TrajectoryObservationKey",
    "TrajectoryObservation",
    "TrajectoryPairDistance",
    "TrajectoryObjectiveResult",
    "TrajectorySeparationObjective",
    "compute_trajectory_objective",
    "normalized_field_distance",
]

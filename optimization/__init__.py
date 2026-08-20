"""Current morphology-search contracts.

The optional Ax dependency is imported only by ``optimization.adapters.ax`` at
the execution boundary; importing this package stays lightweight.
"""
from optimization.design_space import (
    DesignSpace,
    DesignVariable,
    OPTIMIZABLE_PARAMETER_NAMES,
    OptimizableParameterName,
    ParameterSpec,
    PRODUCTION_MAX_TOTAL_PAD_DEPTH_MM,
    PRODUCTION_NOMINAL_VOID_HEIGHT_MM,
    PRODUCTION_SEARCH_BOUNDS,
)
from optimization.evaluation_registry import (
    EvaluationRegistry,
    EvaluationRegistryRecord,
)
from optimization.mechanics_contract import DEFAULT_MECHANICS_CONTRACT, MechanicsContract
from optimization.objectives import (
    OBJECTIVE_NAME,
    TrajectoryObjectiveConfig,
    TrajectoryObservation,
    TrajectoryObjectiveResult,
    compute_trajectory_objective,
    normalized_field_distance,
)
from optimization.protocol import (
    DEFAULT_TRAJECTORY_PROTOCOL,
    PROTOCOL_SCHEMA,
    TrajectoryEvaluationProtocol,
)

__all__ = [
    "DEFAULT_MECHANICS_CONTRACT",
    "DEFAULT_TRAJECTORY_PROTOCOL",
    "DesignSpace",
    "DesignVariable",
    "EvaluationRegistry",
    "EvaluationRegistryRecord",
    "MechanicsContract",
    "OBJECTIVE_NAME",
    "OPTIMIZABLE_PARAMETER_NAMES",
    "OptimizableParameterName",
    "ParameterSpec",
    "PRODUCTION_MAX_TOTAL_PAD_DEPTH_MM",
    "PRODUCTION_NOMINAL_VOID_HEIGHT_MM",
    "PRODUCTION_SEARCH_BOUNDS",
    "PROTOCOL_SCHEMA",
    "TrajectoryEvaluationProtocol",
    "TrajectoryObjectiveConfig",
    "TrajectoryObservation",
    "TrajectoryObjectiveResult",
    "compute_trajectory_objective",
    "normalized_field_distance",
]

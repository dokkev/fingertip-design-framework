"""Algorithm-independent optomechanical design evaluation API."""

from optimization.evaluator import (
    DesignEvaluation,
    DesignEvaluator,
    StateEvaluation,
    TrajectoryEvaluation,
)
from optimization.design_space import (
    DesignSpace,
    DesignVariable,
    OPTIMIZABLE_PARAMETER_NAMES,
    OptimizableParameterName,
)
from optimization.evaluation_registry import (
    EvaluationRegistry,
    EvaluationRegistryRecord,
)
from optimization.scenarios import (
    ContactScenario,
    ScenarioGrid,
    TrajectoryScenario,
)
from optimization.study import (
    OptimizationStudy,
    PRODUCTION_FIXED_FLAT_PAD_WIDTH_MM,
    PRODUCTION_EVALUATION_CONTRACT,
    PRODUCTION_EVALUATION_CONTRACT_ID,
    PRODUCTION_SEARCH_BOUNDS,
    create_production_study,
)

__all__ = [
    "ContactScenario",
    "DesignSpace",
    "DesignEvaluation",
    "DesignEvaluator",
    "DesignVariable",
    "EvaluationRegistry",
    "EvaluationRegistryRecord",
    "OPTIMIZABLE_PARAMETER_NAMES",
    "OptimizationStudy",
    "OptimizableParameterName",
    "PRODUCTION_FIXED_FLAT_PAD_WIDTH_MM",
    "PRODUCTION_EVALUATION_CONTRACT",
    "PRODUCTION_EVALUATION_CONTRACT_ID",
    "PRODUCTION_SEARCH_BOUNDS",
    "StateEvaluation",
    "TrajectoryEvaluation",
    "ScenarioGrid",
    "TrajectoryScenario",
    "create_production_study",
]

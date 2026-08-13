"""Algorithm-independent optomechanical design evaluation API."""

from optimization.evaluator import (
    DesignEvaluation,
    DesignEvaluator,
    PairEvaluation,
    ScenarioEvaluation,
)
from optimization.design_space import (
    DesignSpace,
    DesignVariable,
    OPTIMIZABLE_PARAMETER_NAMES,
    OptimizableParameterName,
)
from optimization.scenarios import (
    ContactScenario,
    ScenarioGrid,
    ScenarioPair,
)
from optimization.study import OptimizationStudy

__all__ = [
    "ContactScenario",
    "DesignSpace",
    "DesignEvaluation",
    "DesignEvaluator",
    "DesignVariable",
    "OPTIMIZABLE_PARAMETER_NAMES",
    "OptimizationStudy",
    "OptimizableParameterName",
    "PairEvaluation",
    "ScenarioEvaluation",
    "ScenarioGrid",
    "ScenarioPair",
]

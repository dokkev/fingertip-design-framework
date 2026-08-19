"""Algorithm-independent optomechanical design evaluation API.

The lightweight design-space and registry contracts are imported eagerly.  The
legacy 2D evaluator/study exports remain available through lazy attribute
resolution so importing the 3D optimization boundary does not initialize
legacy FEM dependencies.
"""

from importlib import import_module

from optimization.design_space import (
    DesignSpace,
    DesignVariable,
    OPTIMIZABLE_PARAMETER_NAMES,
    PRODUCTION_NOMINAL_VOID_HEIGHT_MM,
    PRODUCTION_SEARCH_BOUNDS,
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


_LAZY_EXPORTS = {
    "DesignEvaluation": ("optimization.evaluator", "DesignEvaluation"),
    "DesignEvaluator": ("optimization.evaluator", "DesignEvaluator"),
    "StateEvaluation": ("optimization.evaluator", "StateEvaluation"),
    "TrajectoryEvaluation": ("optimization.evaluator", "TrajectoryEvaluation"),
    "OptimizationStudy": ("optimization.study", "OptimizationStudy"),
    "PRODUCTION_FIXED_FLAT_PAD_WIDTH_MM": (
        "optimization.study",
        "PRODUCTION_FIXED_FLAT_PAD_WIDTH_MM",
    ),
    "PRODUCTION_EVALUATION_CONTRACT": (
        "optimization.study",
        "PRODUCTION_EVALUATION_CONTRACT",
    ),
    "PRODUCTION_EVALUATION_CONTRACT_ID": (
        "optimization.study",
        "PRODUCTION_EVALUATION_CONTRACT_ID",
    ),
    "create_production_study": ("optimization.study", "create_production_study"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value

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
    "PRODUCTION_NOMINAL_VOID_HEIGHT_MM",
    "PRODUCTION_SEARCH_BOUNDS",
    "StateEvaluation",
    "TrajectoryEvaluation",
    "ScenarioGrid",
    "TrajectoryScenario",
    "create_production_study",
]

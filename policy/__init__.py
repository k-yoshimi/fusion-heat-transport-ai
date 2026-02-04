"""Policy module for solver selection and parameter optimization."""

from policy.select import select_best, select_with_ml
from policy.stability import (
    StabilityConstraints,
    SOLVER_STABILITY,
    get_stability,
    is_solver_stable,
    suggest_stable_dt,
)
from policy.optimizer import (
    OptimizationResult,
    ParameterOptimizer,
    optimize_parameters,
)
from policy.physics_selector import (
    PHYSICS_FEATURE_NAMES,
    extract_physics_features,
    PhysicsSolverSelector,
    select_with_physics,
)

__all__ = [
    # Selection
    "select_best",
    "select_with_ml",
    "select_with_physics",
    # Stability
    "StabilityConstraints",
    "SOLVER_STABILITY",
    "get_stability",
    "is_solver_stable",
    "suggest_stable_dt",
    # Optimization
    "OptimizationResult",
    "ParameterOptimizer",
    "optimize_parameters",
    # Physics selector
    "PHYSICS_FEATURE_NAMES",
    "extract_physics_features",
    "PhysicsSolverSelector",
]

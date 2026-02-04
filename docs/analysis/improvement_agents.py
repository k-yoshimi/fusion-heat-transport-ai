"""Multi-agent system for method improvement in solver optimization.

This module provides specialized agents for:
1. Bottleneck identification in Pareto fronts
2. Improvement proposal generation
3. Multi-perspective evaluation
4. Cycle report generation

Architecture:
    BottleneckAnalysisAgent
        └── Identifies gaps, instabilities, performance issues

    ProposalGenerationAgent
        └── Generates parameter_tuning and algorithm_tweak proposals

    EvaluationAgent
        ├── AccuracyPerspective
        ├── SpeedPerspective
        ├── StabilityPerspective
        └── ComplexityPerspective

    ReportAgent
        └── Generates markdown cycle reports
"""

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from docs.analysis.pareto_analyzer import (
    ParetoFront, ParetoPoint, CrossSolverAnalysis, CrossSolverFront,
)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MethodProposal:
    """A proposal for method improvement.

    Attributes:
        proposal_id: Unique identifier (e.g., "P001")
        proposal_type: Type of proposal (new_solver, parameter_tuning, algorithm_tweak)
        title: Short title
        description: Detailed description
        rationale: Why this improvement is needed
        expected_benefit: Expected improvement (qualitative)
        implementation_sketch: Code outline or parameter changes
        status: Current status (proposed, approved, implemented, rejected)
        created_by: Agent that created this proposal
        cycle_id: Improvement cycle number
    """
    proposal_id: str
    proposal_type: str
    title: str
    description: str
    rationale: str
    expected_benefit: str
    implementation_sketch: str
    status: str = "proposed"
    created_by: str = "ProposalGenerationAgent"
    cycle_id: int = 0
    evaluation_scores: Dict[str, float] = field(default_factory=dict)
    evaluation_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MethodProposal":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class Bottleneck:
    """An identified bottleneck in solver performance.

    Attributes:
        bottleneck_id: Unique identifier
        category: Type (stability, accuracy_gap, speed_gap, coverage_gap)
        severity: Level (high, medium, low)
        description: Human-readable description
        affected_solvers: Solvers affected by this bottleneck
        evidence: Supporting data
        suggested_actions: Possible remedies
    """
    bottleneck_id: str
    category: str
    severity: str
    description: str
    affected_solvers: List[str]
    evidence: Dict[str, Any]
    suggested_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


@dataclass
class EvaluationResult:
    """Result of multi-perspective proposal evaluation.

    Attributes:
        proposal_id: ID of evaluated proposal
        scores: Dict mapping perspective to score (0-5)
        overall_score: Weighted average score
        rankings: Position among all proposals
        recommendation: Final recommendation (approve, reject, revise)
        notes: Detailed evaluation notes
    """
    proposal_id: str
    scores: Dict[str, float]
    overall_score: float
    ranking: int = 0
    recommendation: str = "approve"
    notes: List[str] = field(default_factory=list)


# =============================================================================
# Agent Base Class
# =============================================================================

class ImprovementAgent(ABC):
    """Base class for improvement agents."""

    def __init__(self, name: str):
        self.name = name
        self.reasoning_log: List[str] = []

    def log(self, message: str):
        """Log reasoning step."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.reasoning_log.append(f"[{timestamp}] [{self.name}] {message}")

    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> Any:
        """Perform analysis and return results."""
        pass


# =============================================================================
# Bottleneck Analysis Agent
# =============================================================================

class BottleneckAnalysisAgent(ImprovementAgent):
    """Agent for identifying performance bottlenecks.

    Analyzes Pareto fronts to identify:
    - Stability issues (solvers failing at certain parameters)
    - Accuracy gaps (regions where no solver performs well)
    - Speed gaps (accuracy achievable but too slowly)
    - Coverage gaps (parameter ranges not well covered)
    """

    def __init__(self):
        super().__init__("BottleneckAnalysisAgent")

    def analyze(self, context: Dict[str, Any]) -> List[Bottleneck]:
        """Analyze Pareto fronts and cross-solver results for bottlenecks.

        Args:
            context: Dict with:
                - "pareto_fronts": per-solver ParetoFront dict (optional)
                - "cross_solver": CrossSolverAnalysis (optional)

        Returns:
            List of identified bottlenecks
        """
        pareto_fronts = context.get("pareto_fronts", {})
        cross_solver = context.get("cross_solver")

        bottlenecks = []
        self._bottleneck_counter = 0

        # --- Cross-solver analysis (primary) ---
        if cross_solver is not None:
            self.log(f"Analyzing cross-solver results ({len(cross_solver.problems)} problems)...")
            bottlenecks.extend(self._check_cross_solver_gaps(cross_solver))
            bottlenecks.extend(self._check_problem_coverage(cross_solver))
            bottlenecks.extend(self._check_solver_dominance(cross_solver))

        # --- Per-solver analysis (supplementary) ---
        if pareto_fronts:
            self.log(f"Analyzing {len(pareto_fronts)} solvers for per-solver bottlenecks...")
            bottlenecks.extend(self._check_stability(pareto_fronts))
            bottlenecks.extend(self._check_speed_gaps(pareto_fronts))

        if not pareto_fronts and cross_solver is None:
            self.log("No analysis data provided")

        self.log(f"Found {len(bottlenecks)} bottlenecks")
        return bottlenecks

    def _next_bottleneck_id(self) -> str:
        self._bottleneck_counter += 1
        return f"B{self._bottleneck_counter:03d}"

    # -----------------------------------------------------------------
    # Cross-solver bottleneck checks
    # -----------------------------------------------------------------

    def _check_cross_solver_gaps(self, cross: CrossSolverAnalysis) -> List[Bottleneck]:
        """Detect problems where no solver achieves good accuracy."""
        bottlenecks = []

        for gap in cross.coverage_gaps:
            if gap["issue"] == "no_stable_solver":
                bottlenecks.append(Bottleneck(
                    bottleneck_id=self._next_bottleneck_id(),
                    category="no_stable_solver",
                    severity="high",
                    description=gap["description"],
                    affected_solvers=[],
                    evidence={"problem": gap["problem"]},
                    suggested_actions=[
                        "Implement a solver with better stability for this regime",
                        "Add adaptive time-stepping to existing solvers",
                    ],
                ))
                self.log(f"No stable solver for {gap['problem_key']}")

            elif gap["issue"] == "high_error":
                bottlenecks.append(Bottleneck(
                    bottleneck_id=self._next_bottleneck_id(),
                    category="cross_solver_accuracy_gap",
                    severity="high" if gap["best_error"] > 1.0 else "medium",
                    description=gap["description"],
                    affected_solvers=[gap["best_solver"]],
                    evidence={
                        "problem": gap["problem"],
                        "best_error": gap["best_error"],
                        "best_solver": gap["best_solver"],
                    },
                    suggested_actions=[
                        f"Increase resolution for this problem (current best: {gap['best_solver']})",
                        "Consider higher-order scheme for this alpha regime",
                        "New solver may be needed for this problem class",
                    ],
                ))
                self.log(f"High error ({gap['best_error']:.2e}) for {gap['problem_key']}")

        return bottlenecks

    def _check_problem_coverage(self, cross: CrossSolverAnalysis) -> List[Bottleneck]:
        """Check if certain solvers fail on many problems."""
        bottlenecks = []

        for ranking in cross.overall_rankings:
            rate = ranking["stability_rate"]
            if rate < 80:
                bottlenecks.append(Bottleneck(
                    bottleneck_id=self._next_bottleneck_id(),
                    category="solver_instability",
                    severity="high" if rate < 50 else "medium",
                    description=(
                        f"{ranking['solver']} is stable on only "
                        f"{ranking['problems_stable']}/{ranking['problems_total']} "
                        f"problems ({rate:.0f}%)"
                    ),
                    affected_solvers=[ranking["solver"]],
                    evidence={
                        "stability_rate": rate,
                        "problems_stable": ranking["problems_stable"],
                        "problems_total": ranking["problems_total"],
                    },
                    suggested_actions=[
                        f"Reduce dt or add stability constraints for {ranking['solver']}",
                        "Add adaptive time-stepping",
                    ],
                ))
                self.log(f"Solver {ranking['solver']} unstable: {rate:.0f}%")

        return bottlenecks

    def _check_solver_dominance(self, cross: CrossSolverAnalysis) -> List[Bottleneck]:
        """Check if one solver dominates all problems, leaving no niche for others."""
        bottlenecks = []

        wins = cross.solver_win_counts.get("best_accuracy", {})
        total = len(cross.problems)
        if total == 0:
            return bottlenecks

        for solver, count in wins.items():
            ratio = count / total
            if ratio > 0.8:
                other_solvers = [
                    s for s in wins.keys() if s != solver and wins.get(s, 0) == 0
                ]
                if other_solvers:
                    bottlenecks.append(Bottleneck(
                        bottleneck_id=self._next_bottleneck_id(),
                        category="solver_dominance",
                        severity="low",
                        description=(
                            f"{solver} achieves best accuracy in "
                            f"{count}/{total} problems ({ratio*100:.0f}%); "
                            f"{', '.join(other_solvers)} never win"
                        ),
                        affected_solvers=other_solvers,
                        evidence={
                            "dominant_solver": solver,
                            "win_ratio": ratio,
                            "zero_win_solvers": other_solvers,
                        },
                        suggested_actions=[
                            f"Find parameter niches where {', '.join(other_solvers[:3])} can win",
                            "Consider removing unused solvers or improving them",
                        ],
                    ))
                    self.log(f"Dominance: {solver} wins {ratio*100:.0f}%")

        return bottlenecks

    def _check_stability(self, fronts: Dict[str, ParetoFront]) -> List[Bottleneck]:
        """Check for stability issues in each solver."""
        bottlenecks = []
        bottleneck_idx = 1

        for solver_name, front in fronts.items():
            if not front.points:
                continue

            stability_rate = front.summary.get("stability_rate", 100)

            if stability_rate < 90:
                severity = "high" if stability_rate < 50 else "medium"

                # Find unstable configurations
                unstable_configs = [
                    p.config for p in front.points if not p.is_stable
                ]

                # Analyze patterns
                alpha_pattern = self._find_instability_pattern(unstable_configs, "alpha")

                bottlenecks.append(Bottleneck(
                    bottleneck_id=f"B{bottleneck_idx:03d}",
                    category="stability",
                    severity=severity,
                    description=f"{solver_name} has {stability_rate:.1f}% stability rate",
                    affected_solvers=[solver_name],
                    evidence={
                        "stability_rate": stability_rate,
                        "unstable_count": len(unstable_configs),
                        "alpha_pattern": alpha_pattern,
                    },
                    suggested_actions=[
                        f"Reduce dt for {solver_name} at high alpha",
                        f"Implement adaptive time-stepping",
                        f"Add stability check and fallback",
                    ],
                ))
                bottleneck_idx += 1

                self.log(f"Stability issue: {solver_name} at {stability_rate:.1f}%")

        return bottlenecks

    def _find_instability_pattern(
        self, configs: List[Dict], param: str
    ) -> Optional[Dict[str, Any]]:
        """Find patterns in unstable configurations."""
        if not configs:
            return None

        values = [c.get(param) for c in configs if param in c]
        if not values:
            return None

        return {
            "param": param,
            "min_unstable": min(values),
            "max_unstable": max(values),
            "mean_unstable": np.mean(values),
        }

    def _check_accuracy_gaps(self, fronts: Dict[str, ParetoFront]) -> List[Bottleneck]:
        """Check for accuracy gaps in Pareto coverage."""
        bottlenecks = []

        # Find best achievable error across all solvers
        all_errors = []
        for front in fronts.values():
            for p in front.pareto_optimal:
                if p.is_stable and not np.isnan(p.l2_error):
                    all_errors.append((p.l2_error, p.solver))

        if not all_errors:
            return bottlenecks

        best_error = min(e[0] for e in all_errors)
        worst_best_error = max(e[0] for e in all_errors)

        # Check if there's a large gap
        if worst_best_error > best_error * 100:
            underperformers = [
                e[1] for e in all_errors if e[0] > best_error * 10
            ]
            if underperformers:
                bottlenecks.append(Bottleneck(
                    bottleneck_id=f"B{len(bottlenecks)+1:03d}",
                    category="accuracy_gap",
                    severity="medium",
                    description=f"Large accuracy gap: best={best_error:.2e}, worst={worst_best_error:.2e}",
                    affected_solvers=list(set(underperformers)),
                    evidence={
                        "best_error": best_error,
                        "worst_error": worst_best_error,
                        "gap_ratio": worst_best_error / best_error,
                    },
                    suggested_actions=[
                        "Increase spatial resolution for underperformers",
                        "Use higher-order schemes",
                    ],
                ))
                self.log(f"Accuracy gap: {worst_best_error/best_error:.1f}x difference")

        return bottlenecks

    def _check_speed_gaps(self, fronts: Dict[str, ParetoFront]) -> List[Bottleneck]:
        """Check for speed gaps in Pareto coverage."""
        bottlenecks = []

        # Find fastest and slowest solvers
        solver_times = {}
        for solver_name, front in fronts.items():
            times = [p.wall_time for p in front.pareto_optimal if p.is_stable]
            if times:
                solver_times[solver_name] = {
                    "min": min(times),
                    "max": max(times),
                    "mean": np.mean(times),
                }

        if len(solver_times) < 2:
            return bottlenecks

        fastest_mean = min(s["mean"] for s in solver_times.values())
        slowest_mean = max(s["mean"] for s in solver_times.values())

        if slowest_mean > fastest_mean * 100:
            slow_solvers = [
                name for name, stats in solver_times.items()
                if stats["mean"] > fastest_mean * 10
            ]
            if slow_solvers:
                bottlenecks.append(Bottleneck(
                    bottleneck_id=f"B{len(bottlenecks)+1:03d}",
                    category="speed_gap",
                    severity="low",
                    description=f"Speed gap: fastest={fastest_mean*1000:.2f}ms, slowest={slowest_mean*1000:.2f}ms",
                    affected_solvers=slow_solvers,
                    evidence={
                        "fastest_mean": fastest_mean,
                        "slowest_mean": slowest_mean,
                        "gap_ratio": slowest_mean / fastest_mean,
                    },
                    suggested_actions=[
                        "Optimize slow solver implementations",
                        "Use coarser grids where accuracy permits",
                        "Consider parallel implementations",
                    ],
                ))
                self.log(f"Speed gap: {slowest_mean/fastest_mean:.1f}x difference")

        return bottlenecks

    def _check_coverage_gaps(self, fronts: Dict[str, ParetoFront]) -> List[Bottleneck]:
        """Check for gaps in parameter coverage."""
        bottlenecks = []

        # Check if any solver dominates all Pareto-optimal points
        pareto_by_solver = {
            name: len(front.pareto_optimal)
            for name, front in fronts.items()
        }

        total_pareto = sum(pareto_by_solver.values())
        if total_pareto == 0:
            return bottlenecks

        # Check for single-solver dominance
        max_pareto = max(pareto_by_solver.values())
        if max_pareto / total_pareto > 0.8:
            dominant = max(pareto_by_solver.keys(), key=lambda k: pareto_by_solver[k])
            non_dominant = [k for k, v in pareto_by_solver.items() if v < max_pareto * 0.1]

            if non_dominant:
                bottlenecks.append(Bottleneck(
                    bottleneck_id=f"B{len(bottlenecks)+1:03d}",
                    category="coverage_gap",
                    severity="low",
                    description=f"{dominant} dominates {max_pareto/total_pareto*100:.0f}% of Pareto front",
                    affected_solvers=non_dominant,
                    evidence={
                        "dominant_solver": dominant,
                        "dominance_ratio": max_pareto / total_pareto,
                        "pareto_counts": pareto_by_solver,
                    },
                    suggested_actions=[
                        "Tune parameters of non-dominant solvers",
                        "Find niches where other solvers excel",
                    ],
                ))
                self.log(f"Coverage gap: {dominant} dominates {max_pareto/total_pareto*100:.0f}%")

        return bottlenecks


# =============================================================================
# Proposal Generation Agent
# =============================================================================

class ProposalGenerationAgent(ImprovementAgent):
    """Agent for generating improvement proposals.

    Generates proposals based on identified bottlenecks:
    - parameter_tuning: Adjust existing solver parameters
    - algorithm_tweak: Modify solver algorithm
    - new_solver: Propose new solver implementation
    """

    def __init__(self):
        super().__init__("ProposalGenerationAgent")
        self._proposal_counter = 0

    def analyze(self, context: Dict[str, Any]) -> List[MethodProposal]:
        """Generate improvement proposals from bottlenecks.

        Args:
            context: Dict with "bottlenecks" list and "cycle_id"

        Returns:
            List of improvement proposals
        """
        bottlenecks = context.get("bottlenecks", [])
        cycle_id = context.get("cycle_id", 0)

        if not bottlenecks:
            self.log("No bottlenecks provided")
            return []

        self.log(f"Generating proposals for {len(bottlenecks)} bottlenecks...")

        proposals = []

        for bottleneck in bottlenecks:
            new_proposals = self._generate_for_bottleneck(bottleneck, cycle_id)
            proposals.extend(new_proposals)

        self.log(f"Generated {len(proposals)} proposals")
        return proposals

    def _generate_for_bottleneck(
        self, bottleneck: Bottleneck, cycle_id: int
    ) -> List[MethodProposal]:
        """Generate proposals for a specific bottleneck."""
        proposals = []

        if bottleneck.category == "stability":
            proposals.extend(self._stability_proposals(bottleneck, cycle_id))
        elif bottleneck.category == "accuracy_gap":
            proposals.extend(self._accuracy_proposals(bottleneck, cycle_id))
        elif bottleneck.category == "speed_gap":
            proposals.extend(self._speed_proposals(bottleneck, cycle_id))
        elif bottleneck.category == "coverage_gap":
            proposals.extend(self._coverage_proposals(bottleneck, cycle_id))
        elif bottleneck.category == "no_stable_solver":
            proposals.extend(self._no_stable_solver_proposals(bottleneck, cycle_id))
        elif bottleneck.category == "cross_solver_accuracy_gap":
            proposals.extend(self._cross_solver_accuracy_proposals(bottleneck, cycle_id))
        elif bottleneck.category == "solver_instability":
            proposals.extend(self._solver_instability_proposals(bottleneck, cycle_id))
        elif bottleneck.category == "solver_dominance":
            proposals.extend(self._solver_dominance_proposals(bottleneck, cycle_id))

        return proposals

    def _next_proposal_id(self) -> str:
        """Generate next proposal ID."""
        self._proposal_counter += 1
        return f"P{self._proposal_counter:03d}"

    def _stability_proposals(
        self, bottleneck: Bottleneck, cycle_id: int
    ) -> List[MethodProposal]:
        """Generate proposals for stability issues."""
        proposals = []
        solvers = bottleneck.affected_solvers

        # Proposal 1: Adaptive time-stepping
        proposals.append(MethodProposal(
            proposal_id=self._next_proposal_id(),
            proposal_type="algorithm_tweak",
            title=f"Adaptive time-stepping for {', '.join(solvers)}",
            description=(
                f"Implement adaptive dt selection based on local error estimate. "
                f"When |T(n+1) - T_predictor| > tolerance, reduce dt by factor of 2."
            ),
            rationale=(
                f"Stability rate is {bottleneck.evidence.get('stability_rate', 0):.1f}%. "
                f"Adaptive stepping can prevent divergence at challenging parameter regimes."
            ),
            expected_benefit="Improve stability to >95% while maintaining accuracy",
            implementation_sketch="""
# In solver's time-stepping loop:
def adaptive_step(T, T_prev, dt, tol=1e-4):
    # Predictor step
    T_pred = explicit_step(T, dt)
    # Corrector step
    T_corr = implicit_step(T, dt)
    # Error estimate
    error = np.max(np.abs(T_pred - T_corr))
    if error > tol:
        return None, dt / 2  # Reject and reduce dt
    elif error < tol / 10:
        return T_corr, min(dt * 1.5, dt_max)  # Accept and increase dt
    return T_corr, dt  # Accept with same dt
""",
            created_by=self.name,
            cycle_id=cycle_id,
        ))

        # Proposal 2: Parameter limits
        if bottleneck.evidence.get("alpha_pattern"):
            alpha_thresh = bottleneck.evidence["alpha_pattern"].get("min_unstable", 0.5)
            proposals.append(MethodProposal(
                proposal_id=self._next_proposal_id(),
                proposal_type="parameter_tuning",
                title=f"Constrain dt for alpha > {alpha_thresh:.1f}",
                description=(
                    f"For alpha > {alpha_thresh:.1f}, automatically reduce dt by factor "
                    f"proportional to (alpha - {alpha_thresh:.1f})."
                ),
                rationale=(
                    f"Instability pattern shows failures occur when alpha >= {alpha_thresh:.1f}. "
                    f"Smaller dt improves stability at cost of computation time."
                ),
                expected_benefit="Eliminate instability for high-alpha cases",
                implementation_sketch=f"""
# In solver initialization:
def adjust_dt_for_stability(dt, alpha, threshold={alpha_thresh}):
    if alpha > threshold:
        factor = 1.0 / (1 + (alpha - threshold) ** 2)
        return dt * factor
    return dt
""",
                created_by=self.name,
                cycle_id=cycle_id,
            ))

        return proposals

    def _accuracy_proposals(
        self, bottleneck: Bottleneck, cycle_id: int
    ) -> List[MethodProposal]:
        """Generate proposals for accuracy gaps."""
        proposals = []
        gap_ratio = bottleneck.evidence.get("gap_ratio", 1)

        proposals.append(MethodProposal(
            proposal_id=self._next_proposal_id(),
            proposal_type="parameter_tuning",
            title="Increase resolution for underperforming solvers",
            description=(
                f"For solvers with error > {bottleneck.evidence.get('best_error', 0)*10:.2e}, "
                f"automatically increase nr by factor of 1.5 or reduce dt by factor of 2."
            ),
            rationale=f"Accuracy gap is {gap_ratio:.1f}x. Higher resolution reduces discretization error.",
            expected_benefit="Reduce accuracy gap to <10x",
            implementation_sketch="""
# In solver selection:
def select_resolution(solver_name, target_error):
    base_nr, base_dt = default_params[solver_name]
    if estimated_error(solver_name, base_nr, base_dt) > target_error:
        return int(base_nr * 1.5), base_dt / 2
    return base_nr, base_dt
""",
            created_by=self.name,
            cycle_id=cycle_id,
        ))

        return proposals

    def _speed_proposals(
        self, bottleneck: Bottleneck, cycle_id: int
    ) -> List[MethodProposal]:
        """Generate proposals for speed gaps."""
        proposals = []
        slow_solvers = bottleneck.affected_solvers

        proposals.append(MethodProposal(
            proposal_id=self._next_proposal_id(),
            proposal_type="algorithm_tweak",
            title=f"Optimize {', '.join(slow_solvers)} implementation",
            description=(
                "Profile slow solvers and optimize hot paths. "
                "Consider vectorization, caching, or banded matrix solvers."
            ),
            rationale=(
                f"Speed gap is {bottleneck.evidence.get('gap_ratio', 1):.1f}x. "
                "Implementation optimization can reduce time without affecting accuracy."
            ),
            expected_benefit="Reduce computation time by 50%",
            implementation_sketch="""
# Optimization targets:
# 1. Use scipy.linalg.solve_banded for tridiagonal systems
# 2. Pre-compute constant matrices
# 3. Vectorize chi computation
# 4. Use numba @jit for inner loops
""",
            created_by=self.name,
            cycle_id=cycle_id,
        ))

        return proposals

    def _coverage_proposals(
        self, bottleneck: Bottleneck, cycle_id: int
    ) -> List[MethodProposal]:
        """Generate proposals for coverage gaps."""
        proposals = []
        non_dominant = bottleneck.affected_solvers

        if non_dominant:
            proposals.append(MethodProposal(
                proposal_id=self._next_proposal_id(),
                proposal_type="parameter_tuning",
                title=f"Find niches for {', '.join(non_dominant)}",
                description=(
                    "Run targeted parameter sweeps to find configurations where "
                    "non-dominant solvers excel (e.g., specific alpha ranges, IC types)."
                ),
                rationale=(
                    f"Single solver dominates {bottleneck.evidence.get('dominance_ratio', 0)*100:.0f}% "
                    "of Pareto front. Finding niches improves solver diversity."
                ),
                expected_benefit="Each solver has at least 10% Pareto coverage",
                implementation_sketch="""
# Targeted sweep:
niche_configs = {
    'spectral_cosine': {'alpha': [0.0], 'nr': [101, 151]},  # Smooth, high-res
    'compact4_fdm': {'alpha': [0.5, 1.0], 'dt': [0.0001]},  # Nonlinear, small dt
}
""",
                created_by=self.name,
                cycle_id=cycle_id,
            ))

        return proposals

    # -----------------------------------------------------------------
    # Cross-solver bottleneck proposals
    # -----------------------------------------------------------------

    def _no_stable_solver_proposals(
        self, bottleneck: Bottleneck, cycle_id: int
    ) -> List[MethodProposal]:
        """Generate proposals when no solver is stable for a problem."""
        proposals = []
        problem = bottleneck.evidence.get("problem", {})
        alpha = problem.get("alpha", "?")

        proposals.append(MethodProposal(
            proposal_id=self._next_proposal_id(),
            proposal_type="algorithm_tweak",
            title=f"Add adaptive time-stepping for alpha={alpha}",
            description=(
                f"No solver produces a stable result for alpha={alpha}. "
                f"Implement adaptive dt reduction when instability is detected."
            ),
            rationale=bottleneck.description,
            expected_benefit="Enable at least one solver to produce stable results",
            implementation_sketch="""
# Add to SolverBase or specific solver:
def solve_adaptive(self, T0, r, dt, t_end, alpha, max_retries=5):
    for attempt in range(max_retries):
        T_hist = self.solve(T0.copy(), r, dt, t_end, alpha)
        if np.all(np.isfinite(T_hist)) and np.max(np.abs(T_hist)) < 100:
            return T_hist
        dt /= 2  # Halve dt and retry
    return None  # All attempts failed
""",
            created_by=self.name,
            cycle_id=cycle_id,
        ))

        proposals.append(MethodProposal(
            proposal_id=self._next_proposal_id(),
            proposal_type="new_solver",
            title=f"Implement implicit-explicit (IMEX) solver for stiff problems",
            description=(
                f"For high-alpha problems where all solvers fail, "
                f"a dedicated IMEX scheme may handle stiffness better."
            ),
            rationale=bottleneck.description,
            expected_benefit="Stable solutions for previously unsolvable problems",
            implementation_sketch="""
# New solver: ImprovedIMEX with better stiffness handling
# - Treat linear diffusion implicitly (Crank-Nicolson)
# - Treat nonlinear chi term with lagged coefficients
# - Auto-detect stiffness and switch to fully implicit if needed
""",
            created_by=self.name,
            cycle_id=cycle_id,
        ))

        return proposals

    def _cross_solver_accuracy_proposals(
        self, bottleneck: Bottleneck, cycle_id: int
    ) -> List[MethodProposal]:
        """Generate proposals when best cross-solver accuracy is still poor."""
        proposals = []
        best_solver = bottleneck.evidence.get("best_solver", "unknown")
        best_error = bottleneck.evidence.get("best_error", 0)

        proposals.append(MethodProposal(
            proposal_id=self._next_proposal_id(),
            proposal_type="parameter_tuning",
            title=f"Increase resolution for {best_solver}",
            description=(
                f"Best solver ({best_solver}) achieves L2={best_error:.2e}, "
                f"which is above acceptable threshold. Increase nr and decrease dt."
            ),
            rationale=bottleneck.description,
            expected_benefit="Reduce L2 error below 0.1",
            implementation_sketch=f"""
# Extend parameter sweep for {best_solver}:
# Current best config yields L2={best_error:.2e}
# Try: nr in [81, 101, 151], dt in [0.0002, 0.0001]
extended_configs = {{
    'nr': [81, 101, 151],
    'dt': [0.0002, 0.0001],
}}
""",
            created_by=self.name,
            cycle_id=cycle_id,
        ))

        return proposals

    def _solver_instability_proposals(
        self, bottleneck: Bottleneck, cycle_id: int
    ) -> List[MethodProposal]:
        """Generate proposals for solvers that fail on many problems."""
        proposals = []
        solver = bottleneck.affected_solvers[0] if bottleneck.affected_solvers else "unknown"
        rate = bottleneck.evidence.get("stability_rate", 0)

        proposals.append(MethodProposal(
            proposal_id=self._next_proposal_id(),
            proposal_type="parameter_tuning",
            title=f"Restrict dt range for {solver}",
            description=(
                f"{solver} is stable on only {rate:.0f}% of problems. "
                f"Use smaller dt values to improve stability across more problems."
            ),
            rationale=bottleneck.description,
            expected_benefit=f"Improve stability rate of {solver} to >90%",
            implementation_sketch=f"""
# In solver selection / parameter sweep for {solver}:
# Current stability rate: {rate:.0f}%
# Force smaller dt for problematic regimes:
if alpha > 0.5:
    dt = min(dt, 0.0002)
if alpha > 1.0:
    dt = min(dt, 0.0001)
""",
            created_by=self.name,
            cycle_id=cycle_id,
        ))

        return proposals

    def _solver_dominance_proposals(
        self, bottleneck: Bottleneck, cycle_id: int
    ) -> List[MethodProposal]:
        """Generate proposals when one solver dominates all problems."""
        proposals = []
        dominant = bottleneck.evidence.get("dominant_solver", "unknown")
        zero_win = bottleneck.evidence.get("zero_win_solvers", [])

        proposals.append(MethodProposal(
            proposal_id=self._next_proposal_id(),
            proposal_type="parameter_tuning",
            title=f"Tune {', '.join(zero_win[:2])} for niche problems",
            description=(
                f"{dominant} dominates all problem settings. "
                f"Expand parameter ranges for {', '.join(zero_win[:3])} "
                f"to find configurations where they excel."
            ),
            rationale=bottleneck.description,
            expected_benefit="Diversify solver portfolio; each solver wins on at least one problem",
            implementation_sketch=f"""
# Targeted sweep for non-dominant solvers:
# Focus on parameter regimes where {dominant} is weakest
target_configs = {{
    'nr': [101, 151, 201],  # Higher resolution
    'dt': [0.0001, 0.00005],  # Smaller time steps
    'alpha': [0.0, 0.1, 0.2],  # Low-alpha where spectral may excel
}}
""",
            created_by=self.name,
            cycle_id=cycle_id,
        ))

        return proposals


# =============================================================================
# Evaluation Agent
# =============================================================================

class EvaluationPerspective(ABC):
    """Base class for evaluation perspectives."""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def evaluate(self, proposal: MethodProposal, context: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate proposal from this perspective.

        Returns:
            Tuple of (score 0-5, explanation)
        """
        pass


class AccuracyPerspective(EvaluationPerspective):
    """Evaluates proposal's impact on accuracy."""

    def __init__(self):
        super().__init__("accuracy", weight=1.5)

    def evaluate(self, proposal: MethodProposal, context: Dict[str, Any]) -> Tuple[float, str]:
        score = 3.0  # Default neutral
        notes = []

        if "resolution" in proposal.title.lower() or "accuracy" in proposal.expected_benefit.lower():
            score = 4.5
            notes.append("Directly targets accuracy improvement")

        if "adaptive" in proposal.title.lower():
            score = max(score, 4.0)
            notes.append("Adaptive methods can improve accuracy dynamically")

        if proposal.proposal_type == "parameter_tuning":
            # Parameter tuning has moderate accuracy impact
            score = max(score, 3.5)
            notes.append("Parameter tuning can fine-tune accuracy")

        return score, "; ".join(notes) if notes else "Standard accuracy impact"


class SpeedPerspective(EvaluationPerspective):
    """Evaluates proposal's impact on computation speed."""

    def __init__(self):
        super().__init__("speed", weight=1.0)

    def evaluate(self, proposal: MethodProposal, context: Dict[str, Any]) -> Tuple[float, str]:
        score = 3.0
        notes = []

        if "optimize" in proposal.title.lower() or "fast" in proposal.expected_benefit.lower():
            score = 4.5
            notes.append("Directly targets speed improvement")

        if "adaptive" in proposal.title.lower():
            # Adaptive may slow down some cases
            score = 2.5
            notes.append("Adaptive stepping may increase computation for some cases")

        if "resolution" in proposal.title.lower() and "increase" in proposal.title.lower():
            score = 2.0
            notes.append("Higher resolution increases computation time")

        return score, "; ".join(notes) if notes else "Standard speed impact"


class StabilityPerspective(EvaluationPerspective):
    """Evaluates proposal's impact on numerical stability."""

    def __init__(self):
        super().__init__("stability", weight=1.2)

    def evaluate(self, proposal: MethodProposal, context: Dict[str, Any]) -> Tuple[float, str]:
        score = 3.0
        notes = []

        if "stability" in proposal.title.lower() or "adaptive" in proposal.title.lower():
            score = 5.0
            notes.append("Directly addresses stability")

        if "constrain" in proposal.title.lower():
            score = max(score, 4.5)
            notes.append("Parameter constraints improve stability")

        if proposal.proposal_type == "algorithm_tweak":
            score = max(score, 3.5)
            notes.append("Algorithm changes may affect stability")

        return score, "; ".join(notes) if notes else "Standard stability impact"


class ComplexityPerspective(EvaluationPerspective):
    """Evaluates implementation complexity of proposal."""

    def __init__(self):
        super().__init__("complexity", weight=0.8)

    def evaluate(self, proposal: MethodProposal, context: Dict[str, Any]) -> Tuple[float, str]:
        # Higher score = less complex (better)
        score = 3.0
        notes = []

        if proposal.proposal_type == "parameter_tuning":
            score = 4.5
            notes.append("Parameter tuning is simple to implement")

        elif proposal.proposal_type == "algorithm_tweak":
            score = 2.5
            notes.append("Algorithm changes require careful implementation")

        elif proposal.proposal_type == "new_solver":
            score = 1.5
            notes.append("New solver is most complex to implement")

        # Check implementation sketch length
        sketch_lines = len(proposal.implementation_sketch.strip().split("\n"))
        if sketch_lines > 20:
            score = max(1.0, score - 1.0)
            notes.append("Large implementation sketch suggests complexity")

        return score, "; ".join(notes) if notes else "Standard complexity"


class EvaluationAgent(ImprovementAgent):
    """Agent for multi-perspective proposal evaluation.

    Evaluates proposals from multiple perspectives and produces
    an overall ranking for decision-making.
    """

    def __init__(self):
        super().__init__("EvaluationAgent")
        self.perspectives = [
            AccuracyPerspective(),
            SpeedPerspective(),
            StabilityPerspective(),
            ComplexityPerspective(),
        ]

    def analyze(self, context: Dict[str, Any]) -> List[EvaluationResult]:
        """Evaluate all proposals from multiple perspectives.

        Args:
            context: Dict with "proposals" list

        Returns:
            List of EvaluationResult sorted by overall score
        """
        proposals = context.get("proposals", [])
        if not proposals:
            self.log("No proposals to evaluate")
            return []

        self.log(f"Evaluating {len(proposals)} proposals from {len(self.perspectives)} perspectives...")

        results = []
        for proposal in proposals:
            result = self._evaluate_proposal(proposal, context)
            results.append(result)

        # Sort by overall score (descending)
        results.sort(key=lambda r: r.overall_score, reverse=True)

        # Assign rankings
        for i, result in enumerate(results):
            result.ranking = i + 1
            # Set recommendation based on ranking
            if result.overall_score >= 4.0:
                result.recommendation = "approve"
            elif result.overall_score >= 3.0:
                result.recommendation = "consider"
            else:
                result.recommendation = "reject"

        self.log(f"Top proposal: {results[0].proposal_id} with score {results[0].overall_score:.2f}")
        return results

    def _evaluate_proposal(
        self, proposal: MethodProposal, context: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate a single proposal."""
        scores = {}
        notes = []
        total_weight = 0.0
        weighted_sum = 0.0

        for perspective in self.perspectives:
            score, note = perspective.evaluate(proposal, context)
            scores[perspective.name] = score
            notes.append(f"[{perspective.name}] {note}")
            weighted_sum += score * perspective.weight
            total_weight += perspective.weight

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Store scores in proposal for reference
        proposal.evaluation_scores = scores
        proposal.evaluation_notes = notes

        return EvaluationResult(
            proposal_id=proposal.proposal_id,
            scores=scores,
            overall_score=overall_score,
            notes=notes,
        )


# =============================================================================
# Report Agent
# =============================================================================

class ReportAgent(ImprovementAgent):
    """Agent for generating cycle reports.

    Produces markdown reports summarizing:
    - Pareto analysis results
    - Identified bottlenecks
    - Proposals and evaluations
    - Recommendations for next cycle
    """

    def __init__(self):
        super().__init__("ReportAgent")

    def analyze(self, context: Dict[str, Any]) -> str:
        """Generate cycle report.

        Args:
            context: Dict with pareto_fronts, cross_solver, bottlenecks,
                     proposals, evaluations, cycle_id

        Returns:
            Markdown report string
        """
        cycle_id = context.get("cycle_id", 0)
        pareto_fronts = context.get("pareto_fronts", {})
        cross_solver = context.get("cross_solver")
        bottlenecks = context.get("bottlenecks", [])
        proposals = context.get("proposals", [])
        evaluations = context.get("evaluations", [])

        self.log(f"Generating report for cycle {cycle_id}...")

        lines = []
        lines.append(f"# Method Improvement Cycle Report - Cycle {cycle_id}")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Executive Summary
        lines.append("## Executive Summary\n")
        lines.append(f"- **Analyzed solvers:** {len(pareto_fronts)}")
        if cross_solver is not None:
            lines.append(f"- **Problem settings analyzed:** {len(cross_solver.problems)}")
        lines.append(f"- **Identified bottlenecks:** {len(bottlenecks)}")
        lines.append(f"- **Generated proposals:** {len(proposals)}")

        approved = sum(1 for e in evaluations if e.recommendation == "approve")
        lines.append(f"- **Approved proposals:** {approved}")
        lines.append("")

        # Cross-Solver Analysis (primary)
        if cross_solver is not None:
            lines.append("## Cross-Solver Analysis\n")

            # Overall rankings
            if cross_solver.overall_rankings:
                lines.append("### Overall Solver Rankings\n")
                lines.append("| Rank | Solver | Avg Rank | Avg L2 Error | Avg Time (ms) | Stability |")
                lines.append("|------|--------|----------|-------------|---------------|-----------|")
                for i, r in enumerate(cross_solver.overall_rankings):
                    lines.append(
                        f"| {i+1} | {r['solver']} | {r['avg_rank']:.1f} | "
                        f"{r['avg_error']:.2e} | {r['avg_time']*1000:.2f} | "
                        f"{r['stability_rate']:.0f}% |"
                    )
                lines.append("")

            # Win counts
            if cross_solver.solver_win_counts:
                lines.append("### Solver Win Counts\n")
                lines.append("| Solver | Best Accuracy | Best Speed | Pareto-Optimal |")
                lines.append("|--------|---------------|------------|----------------|")
                all_solvers = set()
                for cat in cross_solver.solver_win_counts.values():
                    all_solvers.update(cat.keys())
                for s in sorted(all_solvers):
                    acc = cross_solver.solver_win_counts.get("best_accuracy", {}).get(s, 0)
                    spd = cross_solver.solver_win_counts.get("best_speed", {}).get(s, 0)
                    par = cross_solver.solver_win_counts.get("pareto_optimal", {}).get(s, 0)
                    lines.append(f"| {s} | {acc} | {spd} | {par} |")
                lines.append("")

            # Per-problem results
            lines.append("### Per-Problem Results\n")
            for key, front in cross_solver.problems.items():
                lines.append(f"**{key}**")
                if front.pareto_optimal:
                    pareto_names = [p.solver for p in front.pareto_optimal]
                    lines.append(f"- Pareto-optimal: {', '.join(pareto_names)}")
                    lines.append(f"- Best accuracy: {front.best_by_error}")
                    lines.append(f"- Fastest: {front.best_by_time}")
                else:
                    lines.append("- No stable results")
                lines.append("")

            # Coverage gaps
            if cross_solver.coverage_gaps:
                lines.append("### Coverage Gaps\n")
                for gap in cross_solver.coverage_gaps:
                    lines.append(f"- {gap['description']}")
                lines.append("")

        # Per-Solver Pareto Analysis Results
        lines.append("## Per-Solver Pareto Analysis\n")
        if pareto_fronts:
            lines.append("| Solver | Total Points | Stable | Pareto-Optimal | Min Error | Max Error |")
            lines.append("|--------|--------------|--------|----------------|-----------|-----------|")
            for name, front in pareto_fronts.items():
                s = front.summary
                min_err = f"{s.get('min_error', 0):.2e}" if s.get('min_error') else "N/A"
                max_err = f"{s.get('max_error', 0):.2e}" if s.get('max_error') else "N/A"
                lines.append(
                    f"| {name} | {s.get('total_points', 0)} | "
                    f"{s.get('stable_points', 0)} ({s.get('stability_rate', 0):.0f}%) | "
                    f"{s.get('pareto_optimal_count', 0)} | {min_err} | {max_err} |"
                )
            lines.append("")
        else:
            lines.append("*No per-solver Pareto analysis data available.*\n")

        # Bottlenecks Identified
        lines.append("## Bottlenecks Identified\n")
        if bottlenecks:
            for i, b in enumerate(bottlenecks, 1):
                severity_icon = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}.get(b.severity, "?")
                lines.append(f"### {i}. {b.description}")
                lines.append(f"- **ID:** {b.bottleneck_id}")
                lines.append(f"- **Category:** {b.category}")
                lines.append(f"- **Severity:** {severity_icon}")
                lines.append(f"- **Affected:** {', '.join(b.affected_solvers)}")
                if b.suggested_actions:
                    lines.append("- **Suggested Actions:**")
                    for action in b.suggested_actions:
                        lines.append(f"  - {action}")
                lines.append("")
        else:
            lines.append("*No bottlenecks identified.*\n")

        # Proposals
        lines.append("## Proposals\n")
        if proposals:
            for p in proposals:
                status_icon = {
                    "approved": "APPROVED",
                    "proposed": "PROPOSED",
                    "rejected": "REJECTED",
                }.get(p.status, "?")
                lines.append(f"### {p.proposal_id}: {p.title}")
                lines.append(f"- **Type:** {p.proposal_type}")
                lines.append(f"- **Status:** {status_icon}")
                lines.append(f"- **Rationale:** {p.rationale}")
                lines.append(f"- **Expected Benefit:** {p.expected_benefit}")
                lines.append("")
        else:
            lines.append("*No proposals generated.*\n")

        # Multi-Agent Evaluation
        lines.append("## Multi-Agent Evaluation\n")
        if evaluations:
            lines.append("| Proposal | Accuracy | Speed | Stability | Complexity | Overall | Recommendation |")
            lines.append("|----------|----------|-------|-----------|------------|---------|----------------|")
            for e in evaluations:
                rec_icon = {"approve": "Approve", "consider": "Consider", "reject": "Reject"}.get(
                    e.recommendation, "?"
                )
                lines.append(
                    f"| {e.proposal_id} | "
                    f"{'*' * int(e.scores.get('accuracy', 0))} | "
                    f"{'*' * int(e.scores.get('speed', 0))} | "
                    f"{'*' * int(e.scores.get('stability', 0))} | "
                    f"{'*' * int(e.scores.get('complexity', 0))} | "
                    f"{e.overall_score:.1f} | {rec_icon} |"
                )
            lines.append("")
        else:
            lines.append("*No evaluations available.*\n")

        # Next Cycle Recommendations
        lines.append("## Next Cycle Recommendations\n")
        if evaluations:
            approved_props = [e for e in evaluations if e.recommendation == "approve"]
            if approved_props:
                lines.append("1. **Implement approved proposals:**")
                for e in approved_props:
                    lines.append(f"   - {e.proposal_id}")
            lines.append("2. Re-run Pareto analysis after implementations")
            lines.append("3. Verify improvements meet expected benefits")
        else:
            lines.append("1. Generate more training data for Pareto analysis")
            lines.append("2. Focus on stability improvements")

        lines.append("\n---")
        lines.append("*Report generated by Method Improvement Cycle Framework*")

        report = "\n".join(lines)
        self.log(f"Generated report with {len(lines)} lines")
        return report


# =============================================================================
# Main
# =============================================================================

def main():
    """Demo of improvement agents."""
    print("=" * 60)
    print("IMPROVEMENT AGENTS DEMO")
    print("=" * 60)

    # Create mock Pareto fronts
    mock_fronts = {
        "implicit_fdm": ParetoFront(
            solver_name="implicit_fdm",
            timestamp=datetime.now().isoformat(),
            points=[
                ParetoPoint("implicit_fdm", {"alpha": 0.0, "nr": 51, "dt": 0.001}, 1e-4, 0.01, 0, True),
                ParetoPoint("implicit_fdm", {"alpha": 0.5, "nr": 51, "dt": 0.001}, 2e-4, 0.02, 0, True),
                ParetoPoint("implicit_fdm", {"alpha": 1.0, "nr": 51, "dt": 0.001}, 5e-4, 0.03, 0, True),
            ],
            summary={"stability_rate": 100, "total_points": 3, "stable_points": 3, "pareto_optimal_count": 3},
        ),
        "spectral_cosine": ParetoFront(
            solver_name="spectral_cosine",
            timestamp=datetime.now().isoformat(),
            points=[
                ParetoPoint("spectral_cosine", {"alpha": 0.0, "nr": 51, "dt": 0.001}, 5e-5, 0.005, 0, True),
                ParetoPoint("spectral_cosine", {"alpha": 0.5, "nr": 51, "dt": 0.001}, float("nan"), 0.0, 0, False),
                ParetoPoint("spectral_cosine", {"alpha": 1.0, "nr": 51, "dt": 0.001}, float("nan"), 0.0, 0, False),
            ],
            summary={"stability_rate": 33.3, "total_points": 3, "stable_points": 1, "pareto_optimal_count": 1},
        ),
    }
    mock_fronts["implicit_fdm"].pareto_optimal = mock_fronts["implicit_fdm"].points
    mock_fronts["spectral_cosine"].pareto_optimal = [mock_fronts["spectral_cosine"].points[0]]

    # Run bottleneck analysis
    print("\n[1] Bottleneck Analysis")
    print("-" * 40)
    bottleneck_agent = BottleneckAnalysisAgent()
    bottlenecks = bottleneck_agent.analyze({"pareto_fronts": mock_fronts})
    for b in bottlenecks:
        print(f"  {b.bottleneck_id}: [{b.severity}] {b.description}")

    # Generate proposals
    print("\n[2] Proposal Generation")
    print("-" * 40)
    proposal_agent = ProposalGenerationAgent()
    proposals = proposal_agent.analyze({"bottlenecks": bottlenecks, "cycle_id": 1})
    for p in proposals:
        print(f"  {p.proposal_id}: {p.title}")

    # Evaluate proposals
    print("\n[3] Multi-Perspective Evaluation")
    print("-" * 40)
    eval_agent = EvaluationAgent()
    evaluations = eval_agent.analyze({"proposals": proposals})
    for e in evaluations:
        print(f"  #{e.ranking} {e.proposal_id}: {e.overall_score:.2f} ({e.recommendation})")

    # Generate report
    print("\n[4] Report Generation")
    print("-" * 40)
    report_agent = ReportAgent()
    report = report_agent.analyze({
        "pareto_fronts": mock_fronts,
        "bottlenecks": bottlenecks,
        "proposals": proposals,
        "evaluations": evaluations,
        "cycle_id": 1,
    })
    print(f"  Generated {len(report)} character report")

    # Save report
    report_path = os.path.join(PROJECT_ROOT, "data", "improvement_cycle_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved to: {report_path}")


if __name__ == "__main__":
    main()

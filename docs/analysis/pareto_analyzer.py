"""Pareto analysis for solver performance optimization.

This module provides tools for computing Pareto fronts across
solver configurations, analyzing trade-offs between accuracy and speed.

Two levels of analysis:
    1. Per-solver: How each solver performs across (nr, dt) for a given problem
    2. Cross-solver: For each problem, which solver is Pareto-optimal

Key Components:
    - ParetoPoint: Single configuration result
    - ParetoFront: Per-solver collection of points
    - CrossSolverFront: Cross-solver comparison for a fixed problem
    - CrossSolverAnalysis: Full cross-solver analysis across problems
    - ParetoAnalysisAgent: Runs parameter sweeps and computes fronts
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from solvers.base import SolverBase
from metrics.accuracy import compute_errors


def _has_physbo() -> bool:
    """Check if physbo is available."""
    try:
        import physbo  # noqa: F401
        return True
    except ImportError:
        return False


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ParetoPoint:
    """A single point in the Pareto analysis space.

    Attributes:
        solver: Name of the solver
        config: Configuration dict {alpha, nr, dt, t_end, ic_type}
        l2_error: L2 error vs reference
        wall_time: Computation time in seconds
        pareto_rank: Rank in Pareto front (0 = Pareto-optimal)
        is_stable: Whether the solution is numerically stable
    """
    solver: str
    config: Dict[str, Any]
    l2_error: float
    wall_time: float
    pareto_rank: int = 0
    is_stable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "solver": self.solver,
            "config": self.config,
            "l2_error": self.l2_error if not np.isnan(self.l2_error) else None,
            "wall_time": self.wall_time,
            "pareto_rank": self.pareto_rank,
            "is_stable": self.is_stable,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParetoPoint":
        """Create from dictionary."""
        return cls(
            solver=d["solver"],
            config=d["config"],
            l2_error=d["l2_error"] if d["l2_error"] is not None else float("nan"),
            wall_time=d["wall_time"],
            pareto_rank=d.get("pareto_rank", 0),
            is_stable=d.get("is_stable", True),
        )


@dataclass
class ParetoFront:
    """Collection of Pareto analysis points for a solver.

    Attributes:
        solver_name: Name of the solver
        timestamp: When the analysis was performed
        points: All points in the analysis
        pareto_optimal: Points with pareto_rank == 0
        summary: Summary statistics
    """
    solver_name: str
    timestamp: str
    points: List[ParetoPoint] = field(default_factory=list)
    pareto_optimal: List[ParetoPoint] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def compute_pareto_optimal(self):
        """Compute Pareto-optimal subset from points."""
        if not self.points:
            return

        # Filter stable points only
        stable_points = [p for p in self.points if p.is_stable]
        if not stable_points:
            self.pareto_optimal = []
            return

        # Compute Pareto ranks
        error_time_pairs = [(p.l2_error, p.wall_time) for p in stable_points]
        for i, point in enumerate(stable_points):
            others = [error_time_pairs[j] for j in range(len(stable_points)) if j != i]
            is_dominated = _is_pareto_dominated(
                (point.l2_error, point.wall_time), others
            )
            point.pareto_rank = 1 if is_dominated else 0

        self.pareto_optimal = [p for p in stable_points if p.pareto_rank == 0]

    def compute_summary(self):
        """Compute summary statistics."""
        stable_points = [p for p in self.points if p.is_stable]

        if not stable_points:
            self.summary = {
                "total_points": len(self.points),
                "stable_points": 0,
                "pareto_optimal_count": 0,
                "stability_rate": 0.0,
            }
            return

        errors = [p.l2_error for p in stable_points if not np.isnan(p.l2_error)]
        times = [p.wall_time for p in stable_points]

        self.summary = {
            "total_points": len(self.points),
            "stable_points": len(stable_points),
            "pareto_optimal_count": len(self.pareto_optimal),
            "stability_rate": len(stable_points) / len(self.points) * 100,
            "min_error": min(errors) if errors else None,
            "max_error": max(errors) if errors else None,
            "min_time": min(times) if times else None,
            "max_time": max(times) if times else None,
            "error_range_log10": (
                np.log10(max(errors)) - np.log10(min(errors))
                if errors and min(errors) > 0 else None
            ),
            "time_range_log10": (
                np.log10(max(times)) - np.log10(min(times))
                if times and min(times) > 0 else None
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "solver_name": self.solver_name,
            "timestamp": self.timestamp,
            "points": [p.to_dict() for p in self.points],
            "pareto_optimal": [p.to_dict() for p in self.pareto_optimal],
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParetoFront":
        """Create from dictionary."""
        front = cls(
            solver_name=d["solver_name"],
            timestamp=d["timestamp"],
            points=[ParetoPoint.from_dict(p) for p in d.get("points", [])],
            pareto_optimal=[ParetoPoint.from_dict(p) for p in d.get("pareto_optimal", [])],
            summary=d.get("summary", {}),
        )
        return front

    def save(self, path: str):
        """Save to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ParetoFront":
        """Load from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Cross-Solver Data Structures
# =============================================================================

@dataclass
class CrossSolverFront:
    """Pareto front across solvers for a fixed problem setting.

    For a given (alpha, ic_type), this compares all solvers at their
    best (nr, dt) configurations.

    Attributes:
        problem_key: String key like "alpha=0.5_ic=parabola"
        problem: Problem description dict {alpha, ic_type}
        points: Best points from each solver
        pareto_optimal: Cross-solver Pareto-optimal points
        best_by_error: Solver with lowest error
        best_by_time: Solver with fastest time
        solver_rankings: Ranking of solvers by combined score
    """
    problem_key: str
    problem: Dict[str, Any]
    points: List[ParetoPoint] = field(default_factory=list)
    pareto_optimal: List[ParetoPoint] = field(default_factory=list)
    best_by_error: str = ""
    best_by_time: str = ""
    solver_rankings: List[Dict[str, Any]] = field(default_factory=list)

    def compute(self):
        """Compute Pareto-optimal subset and rankings from points."""
        stable = [p for p in self.points if p.is_stable and not np.isnan(p.l2_error)]
        if not stable:
            return

        # Pareto-optimal across solvers
        pairs = [(p.l2_error, p.wall_time) for p in stable]
        for i, point in enumerate(stable):
            others = [pairs[j] for j in range(len(stable)) if j != i]
            point.pareto_rank = 1 if _is_pareto_dominated(pairs[i], others) else 0

        self.pareto_optimal = [p for p in stable if p.pareto_rank == 0]

        # Best by single objective
        best_err = min(stable, key=lambda p: p.l2_error)
        self.best_by_error = best_err.solver
        best_time = min(stable, key=lambda p: p.wall_time)
        self.best_by_time = best_time.solver

        # Rank all solvers by score = error + 0.1 * time
        ranked = sorted(stable, key=lambda p: p.l2_error + 0.1 * p.wall_time)
        self.solver_rankings = [
            {
                "rank": i + 1,
                "solver": p.solver,
                "l2_error": p.l2_error,
                "wall_time": p.wall_time,
                "config": p.config,
                "is_pareto_optimal": p.pareto_rank == 0,
            }
            for i, p in enumerate(ranked)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_key": self.problem_key,
            "problem": self.problem,
            "points": [p.to_dict() for p in self.points],
            "pareto_optimal": [p.to_dict() for p in self.pareto_optimal],
            "best_by_error": self.best_by_error,
            "best_by_time": self.best_by_time,
            "solver_rankings": self.solver_rankings,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CrossSolverFront":
        return cls(
            problem_key=d["problem_key"],
            problem=d["problem"],
            points=[ParetoPoint.from_dict(p) for p in d.get("points", [])],
            pareto_optimal=[ParetoPoint.from_dict(p) for p in d.get("pareto_optimal", [])],
            best_by_error=d.get("best_by_error", ""),
            best_by_time=d.get("best_by_time", ""),
            solver_rankings=d.get("solver_rankings", []),
        )


@dataclass
class CrossSolverAnalysis:
    """Full cross-solver analysis across all problem settings.

    Attributes:
        timestamp: When the analysis was performed
        problems: Dict mapping problem_key to CrossSolverFront
        solver_win_counts: How many problems each solver wins
        overall_rankings: Overall solver rankings aggregated
        coverage_gaps: Problem settings with poor solver coverage
    """
    timestamp: str = ""
    problems: Dict[str, CrossSolverFront] = field(default_factory=dict)
    solver_win_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    overall_rankings: List[Dict[str, Any]] = field(default_factory=list)
    coverage_gaps: List[Dict[str, Any]] = field(default_factory=list)

    def compute_summary(self):
        """Compute aggregate statistics across all problems."""
        if not self.problems:
            return

        # Count wins by solver
        wins_by_error: Dict[str, int] = {}
        wins_by_time: Dict[str, int] = {}
        wins_by_pareto: Dict[str, int] = {}

        for front in self.problems.values():
            if front.best_by_error:
                wins_by_error[front.best_by_error] = wins_by_error.get(front.best_by_error, 0) + 1
            if front.best_by_time:
                wins_by_time[front.best_by_time] = wins_by_time.get(front.best_by_time, 0) + 1
            for p in front.pareto_optimal:
                wins_by_pareto[p.solver] = wins_by_pareto.get(p.solver, 0) + 1

        self.solver_win_counts = {
            "best_accuracy": wins_by_error,
            "best_speed": wins_by_time,
            "pareto_optimal": wins_by_pareto,
        }

        # Overall rankings: aggregate score across problems
        solver_scores: Dict[str, List[float]] = {}
        solver_errors: Dict[str, List[float]] = {}
        solver_times: Dict[str, List[float]] = {}
        solver_stable_count: Dict[str, int] = {}
        total_problems = len(self.problems)

        for front in self.problems.values():
            for r in front.solver_rankings:
                name = r["solver"]
                if name not in solver_scores:
                    solver_scores[name] = []
                    solver_errors[name] = []
                    solver_times[name] = []
                    solver_stable_count[name] = 0
                solver_scores[name].append(r["rank"])
                solver_errors[name].append(r["l2_error"])
                solver_times[name].append(r["wall_time"])
                solver_stable_count[name] += 1

        self.overall_rankings = sorted(
            [
                {
                    "solver": name,
                    "avg_rank": np.mean(ranks),
                    "avg_error": np.mean(solver_errors[name]),
                    "avg_time": np.mean(solver_times[name]),
                    "problems_stable": solver_stable_count[name],
                    "problems_total": total_problems,
                    "stability_rate": solver_stable_count[name] / total_problems * 100,
                }
                for name, ranks in solver_scores.items()
            ],
            key=lambda x: x["avg_rank"],
        )

        # Detect coverage gaps: problems where best error is still high
        self.coverage_gaps = []
        for key, front in self.problems.items():
            stable = [p for p in front.points if p.is_stable and not np.isnan(p.l2_error)]
            if not stable:
                self.coverage_gaps.append({
                    "problem_key": key,
                    "problem": front.problem,
                    "issue": "no_stable_solver",
                    "description": f"No solver produces stable result for {key}",
                })
            else:
                best = min(p.l2_error for p in stable)
                if best > 0.5:
                    self.coverage_gaps.append({
                        "problem_key": key,
                        "problem": front.problem,
                        "issue": "high_error",
                        "best_error": best,
                        "best_solver": min(stable, key=lambda p: p.l2_error).solver,
                        "description": f"Best L2 error for {key} is {best:.2e} (solver: {min(stable, key=lambda p: p.l2_error).solver})",
                    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "problems": {k: v.to_dict() for k, v in self.problems.items()},
            "solver_win_counts": self.solver_win_counts,
            "overall_rankings": self.overall_rankings,
            "coverage_gaps": self.coverage_gaps,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CrossSolverAnalysis":
        return cls(
            timestamp=d.get("timestamp", ""),
            problems={k: CrossSolverFront.from_dict(v) for k, v in d.get("problems", {}).items()},
            solver_win_counts=d.get("solver_win_counts", {}),
            overall_rankings=d.get("overall_rankings", []),
            coverage_gaps=d.get("coverage_gaps", []),
        )

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CrossSolverAnalysis":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Helper Functions
# =============================================================================

def _is_pareto_dominated(
    candidate: Tuple[float, float],
    others: List[Tuple[float, float]],
) -> bool:
    """Check if candidate is Pareto-dominated by any other point.

    A point is dominated if another point has lower or equal values
    in all objectives, and strictly lower in at least one.

    Args:
        candidate: (error, time) tuple
        others: List of (error, time) tuples

    Returns:
        True if dominated
    """
    for other in others:
        if other[0] <= candidate[0] and other[1] <= candidate[1]:
            if other[0] < candidate[0] or other[1] < candidate[1]:
                return True
    return False


def compute_pareto_ranks(points: List[Tuple[float, float]]) -> List[int]:
    """Compute Pareto ranks for a list of points.

    Rank 0 = Pareto-optimal (non-dominated)
    Rank 1+ = dominated by lower-rank points

    Args:
        points: List of (error, time) tuples

    Returns:
        List of ranks (same order as input)
    """
    n = len(points)
    ranks = [0] * n

    for i in range(n):
        others = [points[j] for j in range(n) if j != i]
        if _is_pareto_dominated(points[i], others):
            ranks[i] = 1

    return ranks


# =============================================================================
# Initial Condition Factory
# =============================================================================

def make_initial(r: np.ndarray, ic_type: str, ic_scale: float = 1.0) -> np.ndarray:
    """Create initial condition based on type.

    Args:
        r: Radial grid
        ic_type: Type of initial condition
        ic_scale: Scaling factor

    Returns:
        Initial temperature profile
    """
    if ic_type == "parabola":
        return ic_scale * (1 - r**2)
    elif ic_type == "gaussian":
        return ic_scale * np.exp(-10 * r**2)
    elif ic_type == "cosine":
        return ic_scale * np.cos(np.pi * r / 2)
    elif ic_type == "sine":
        return ic_scale * np.sin(np.pi * (1 - r))
    elif ic_type == "step":
        return ic_scale * np.where(r < 0.3, 1.0, 0.0)
    else:
        raise ValueError(f"Unknown IC type: {ic_type}")


# =============================================================================
# PHYSBO Simulator
# =============================================================================

class _SolverSimulator:
    """PHYSBO-compatible simulator for solver evaluation.

    Evaluates (solver, config) pairs and returns negated objectives
    (PHYSBO maximizes, so we negate for minimization).

    Attributes:
        evaluated_points: Cache of action_index -> ParetoPoint
    """

    PENALTY = -100.0

    def __init__(self, agent: "ParetoAnalysisAgent", solver: SolverBase,
                 alpha: float, ic_type: str, nr: int, t_end: float,
                 dt_candidates: np.ndarray):
        self._agent = agent
        self._solver = solver
        self._alpha = alpha
        self._ic_type = ic_type
        self._nr = nr
        self._t_end = t_end
        self._dt_candidates = dt_candidates
        self.evaluated_points: Dict[int, ParetoPoint] = {}

        # Pre-compute all objective values (N, 2) for PHYSBO
        self.t = np.full((len(dt_candidates), 2), self.PENALTY)

    def __call__(self, action):
        """Evaluate objectives for given action indices.

        Args:
            action: Array of candidate indices

        Returns:
            (N, 2) array of negated [L2_error, wall_time]
        """
        for idx in action:
            if idx in self.evaluated_points:
                continue
            dt = float(self._dt_candidates[idx])
            config = {
                "alpha": self._alpha,
                "nr": self._nr,
                "dt": dt,
                "t_end": self._t_end,
                "ic_type": self._ic_type,
            }
            point = self._agent._run_single_config(self._solver, config)
            self.evaluated_points[idx] = point

            if point.is_stable and not np.isnan(point.l2_error):
                self.t[idx, 0] = -point.l2_error
                self.t[idx, 1] = -point.wall_time
            # else: stays at PENALTY

        return self.t[action]


# =============================================================================
# Pareto Analysis Agent
# =============================================================================

class ParetoAnalysisAgent:
    """Agent for performing Pareto analysis on solver performance.

    Runs parameter sweeps across configurations and computes
    Pareto fronts showing trade-offs between accuracy and speed.
    """

    def __init__(
        self,
        output_dir: str = None,
        alpha_list: List[float] = None,
        nr_list: List[int] = None,
        dt_list: List[float] = None,
        t_end_list: List[float] = None,
        ic_types: List[str] = None,
        use_physbo: bool = None,
        fixed_nr: int = 61,
        physbo_n_candidates: int = 80,
        physbo_n_random: int = 5,
        physbo_n_bayes: int = 15,
        physbo_score: str = "HVPI",
    ):
        """Initialize the Pareto analysis agent.

        Args:
            output_dir: Directory to save Pareto front data
            alpha_list: Alpha values to sweep
            nr_list: Grid sizes to sweep (deprecated, use fixed_nr)
            dt_list: Time steps to sweep
            t_end_list: End times to sweep
            ic_types: Initial condition types to sweep
            use_physbo: Whether to use PHYSBO (None = auto-detect)
            fixed_nr: Fixed grid size for both PHYSBO and grid sweep
            physbo_n_candidates: Number of dt candidates in log space
            physbo_n_random: Number of random search steps
            physbo_n_bayes: Number of Bayesian optimization steps
            physbo_score: Acquisition function for PHYSBO
        """
        self.output_dir = output_dir or os.path.join(
            PROJECT_ROOT, "data", "pareto_fronts"
        )
        self.alpha_list = alpha_list or [0.0, 0.2, 0.5, 1.0]
        self.nr_list = nr_list or [31, 51, 71]
        self.dt_list = dt_list or [0.002, 0.001, 0.0005, 0.0002]
        self.t_end_list = t_end_list or [0.1]
        self.ic_types = ic_types or ["parabola"]

        # Fixed grid size for analysis
        self._fixed_nr = fixed_nr

        # PHYSBO settings
        if use_physbo is None:
            self._use_physbo = _has_physbo()
        else:
            self._use_physbo = use_physbo
        self._physbo_n_candidates = physbo_n_candidates
        self._physbo_n_random = physbo_n_random
        self._physbo_n_bayes = physbo_n_bayes
        self._physbo_score = physbo_score

        # Reference solver for error computation
        self._ref_solver = None

    def _get_reference_solver(self):
        """Get reference solver instance."""
        if self._ref_solver is None:
            from solvers.fdm.implicit import ImplicitFDM
            self._ref_solver = ImplicitFDM()
        return self._ref_solver

    def _compute_reference(
        self, T0: np.ndarray, r: np.ndarray, dt: float, t_end: float, alpha: float
    ) -> np.ndarray:
        """Compute reference solution with 4x refinement.

        Args:
            T0: Initial temperature
            r: Radial grid
            dt: Time step
            t_end: End time
            alpha: Nonlinearity parameter

        Returns:
            Reference solution history
        """
        nr_fine = 4 * len(r) - 3
        r_fine = np.linspace(0, 1, nr_fine)
        T0_fine = np.interp(r_fine, r, T0)

        solver = self._get_reference_solver()
        T_hist = solver.solve(T0_fine, r_fine, dt / 4, t_end, alpha)

        # Downsample back to original grid
        indices = np.linspace(0, nr_fine - 1, len(r)).astype(int)
        return T_hist[:, indices]

    def _build_dt_candidates(self, n: int = None) -> np.ndarray:
        """Build log-spaced dt candidates for PHYSBO search.

        Args:
            n: Number of candidates (default: physbo_n_candidates)

        Returns:
            Array of dt values in [1e-5, 1e-2]
        """
        if n is None:
            n = self._physbo_n_candidates
        return np.logspace(-5, -2, n)

    def _find_best_for_problem_physbo(
        self,
        solver: SolverBase,
        alpha: float,
        ic_type: str,
    ) -> Optional[ParetoPoint]:
        """Find best dt for a solver using PHYSBO multi-objective optimization.

        Args:
            solver: Solver instance
            alpha: Nonlinearity parameter
            ic_type: Initial condition type

        Returns:
            Best ParetoPoint (min L2 error among stable) or None
        """
        import physbo

        dt_candidates = self._build_dt_candidates()
        t_end = self.t_end_list[0] if self.t_end_list else 0.1

        simulator = _SolverSimulator(
            agent=self,
            solver=solver,
            alpha=alpha,
            ic_type=ic_type,
            nr=self._fixed_nr,
            t_end=t_end,
            dt_candidates=dt_candidates,
        )

        # Feature matrix: log10(dt)
        test_X = np.log10(dt_candidates).reshape(-1, 1)

        policy = physbo.search.discrete_multi.Policy(
            test_X=test_X, num_objectives=2
        )
        policy.set_seed(42)

        policy.random_search(
            max_num_probes=self._physbo_n_random,
            simulator=simulator,
            is_disp=False,
        )
        policy.bayes_search(
            max_num_probes=self._physbo_n_bayes,
            simulator=simulator,
            score=self._physbo_score,
            interval=0,
            is_disp=False,
        )

        # Find best stable point from evaluated cache
        best_point = None
        for point in simulator.evaluated_points.values():
            if point.is_stable and not np.isnan(point.l2_error):
                if best_point is None or point.l2_error < best_point.l2_error:
                    best_point = point

        # Fallback: return any evaluated point if no stable found
        if best_point is None and simulator.evaluated_points:
            best_point = next(iter(simulator.evaluated_points.values()))

        return best_point

    def analyze_solver(
        self,
        solver: SolverBase,
        verbose: bool = True,
    ) -> ParetoFront:
        """Perform Pareto analysis for a single solver.

        Dispatches to PHYSBO-based analysis if enabled, otherwise
        falls back to grid sweep.

        Args:
            solver: Solver instance to analyze
            verbose: Whether to print progress

        Returns:
            ParetoFront with analysis results
        """
        if self._use_physbo:
            try:
                return self._analyze_solver_physbo(solver, verbose)
            except Exception:
                if verbose:
                    print(f"  PHYSBO failed for {solver.name}, falling back to grid sweep")

        return self._analyze_solver_grid(solver, verbose)

    def _analyze_solver_physbo(
        self,
        solver: SolverBase,
        verbose: bool = True,
    ) -> ParetoFront:
        """Perform Pareto analysis using PHYSBO for dt exploration.

        For each (alpha, ic_type) problem, runs PHYSBO to search dt
        space with fixed nr.

        Args:
            solver: Solver instance to analyze
            verbose: Whether to print progress

        Returns:
            ParetoFront with analysis results
        """
        timestamp = datetime.now().isoformat()
        front = ParetoFront(
            solver_name=solver.name,
            timestamp=timestamp,
        )

        problems = [
            (alpha, ic_type)
            for alpha in self.alpha_list
            for ic_type in self.ic_types
        ]

        if verbose:
            print(f"\nAnalyzing {solver.name} via PHYSBO ({len(problems)} problems, "
                  f"{self._physbo_n_random}+{self._physbo_n_bayes} probes each)...")

        for alpha, ic_type in problems:
            if verbose:
                print(f"  alpha={alpha}, ic={ic_type}...", end="", flush=True)

            point = self._find_best_for_problem_physbo(solver, alpha, ic_type)
            if point is not None:
                front.points.append(point)
                if verbose:
                    if point.is_stable and not np.isnan(point.l2_error):
                        print(f" L2={point.l2_error:.2e}, dt={point.config['dt']:.1e}")
                    else:
                        print(" UNSTABLE")
            elif verbose:
                print(" no result")

        front.compute_pareto_optimal()
        front.compute_summary()

        if verbose:
            print(f"  Completed: {front.summary['stable_points']}/{front.summary['total_points']} stable")
            print(f"  Pareto-optimal: {front.summary['pareto_optimal_count']} points")

        return front

    def _analyze_solver_grid(
        self,
        solver: SolverBase,
        verbose: bool = True,
    ) -> ParetoFront:
        """Perform Pareto analysis using grid sweep with fixed nr.

        Args:
            solver: Solver instance to analyze
            verbose: Whether to print progress

        Returns:
            ParetoFront with analysis results
        """
        timestamp = datetime.now().isoformat()
        front = ParetoFront(
            solver_name=solver.name,
            timestamp=timestamp,
        )

        nr = self._fixed_nr

        # Total configurations
        total = (
            len(self.alpha_list) *
            len(self.dt_list) * len(self.t_end_list) * len(self.ic_types)
        )
        count = 0

        if verbose:
            print(f"\nAnalyzing {solver.name} (nr={nr}, {total} configurations)...")

        for alpha in self.alpha_list:
            for dt in self.dt_list:
                for t_end in self.t_end_list:
                    for ic_type in self.ic_types:
                        count += 1
                        if verbose and count % 20 == 0:
                            print(f"  Progress: {count}/{total} ({100*count/total:.0f}%)")

                        config = {
                            "alpha": alpha,
                            "nr": nr,
                            "dt": dt,
                            "t_end": t_end,
                            "ic_type": ic_type,
                        }

                        point = self._run_single_config(solver, config)
                        front.points.append(point)

        # Compute Pareto-optimal subset and summary
        front.compute_pareto_optimal()
        front.compute_summary()

        if verbose:
            print(f"  Completed: {front.summary['stable_points']}/{front.summary['total_points']} stable")
            print(f"  Pareto-optimal: {front.summary['pareto_optimal_count']} points")

        return front

    def _run_single_config(
        self,
        solver: SolverBase,
        config: Dict[str, Any],
    ) -> ParetoPoint:
        """Run solver with a single configuration.

        Args:
            solver: Solver instance
            config: Configuration dict

        Returns:
            ParetoPoint with results
        """
        alpha = config["alpha"]
        nr = config["nr"]
        dt = config["dt"]
        t_end = config["t_end"]
        ic_type = config["ic_type"]

        r = np.linspace(0, 1, nr)
        T0 = make_initial(r, ic_type)

        # Run solver
        try:
            t0 = time.perf_counter()
            T_hist = solver.solve(T0.copy(), r, dt, t_end, alpha)
            wall_time = time.perf_counter() - t0

            # Check stability
            is_nan = bool(np.any(np.isnan(T_hist)))
            max_T = float(np.max(np.abs(T_hist))) if not is_nan else float("inf")
            is_stable = not is_nan and max_T < 100

            if is_stable:
                # Compute reference and errors
                T_ref = self._compute_reference(T0, r, dt, t_end, alpha)
                nt_ref, nt_sol = T_ref.shape[0], T_hist.shape[0]

                if nt_sol != nt_ref:
                    T_cmp = np.stack([T_hist[0], T_hist[-1]])
                    T_ref_cmp = np.stack([T_ref[0], T_ref[-1]])
                else:
                    T_cmp, T_ref_cmp = T_hist, T_ref

                errs = compute_errors(T_cmp, T_ref_cmp, r)
                l2_error = errs["l2"]
            else:
                l2_error = float("nan")

        except Exception:
            wall_time = 0.0
            l2_error = float("nan")
            is_stable = False

        return ParetoPoint(
            solver=solver.name,
            config=config,
            l2_error=l2_error,
            wall_time=wall_time,
            pareto_rank=0,
            is_stable=is_stable,
        )

    def analyze_all_solvers(
        self,
        solvers: List[SolverBase],
        verbose: bool = True,
    ) -> Dict[str, ParetoFront]:
        """Analyze all solvers and return Pareto fronts.

        Args:
            solvers: List of solver instances
            verbose: Whether to print progress

        Returns:
            Dict mapping solver name to ParetoFront
        """
        results = {}

        for solver in solvers:
            front = self.analyze_solver(solver, verbose)
            results[solver.name] = front

            # Save to file
            filename = f"{solver.name}_{front.timestamp.replace(':', '-')}.json"
            path = os.path.join(self.output_dir, filename)
            front.save(path)

            if verbose:
                print(f"  Saved to: {path}")

        return results

    def run_quick_analysis(
        self,
        solvers: List[SolverBase],
        verbose: bool = True,
    ) -> Dict[str, ParetoFront]:
        """Run quick analysis with reduced parameter space.

        Args:
            solvers: List of solver instances
            verbose: Whether to print progress

        Returns:
            Dict mapping solver name to ParetoFront
        """
        # Store original values
        orig_alpha = self.alpha_list
        orig_dt = self.dt_list

        # Use reduced sweep
        self.alpha_list = [0.0, 0.5, 1.0]
        self.dt_list = [0.001, 0.0005]

        try:
            return self.analyze_all_solvers(solvers, verbose)
        finally:
            # Restore original values
            self.alpha_list = orig_alpha
            self.dt_list = orig_dt

    # -----------------------------------------------------------------
    # Cross-solver analysis
    # -----------------------------------------------------------------

    def analyze_cross_solver(
        self,
        solvers: List[SolverBase],
        verbose: bool = True,
    ) -> CrossSolverAnalysis:
        """Run cross-solver Pareto analysis.

        For each problem setting (alpha, ic_type), runs all solvers
        across (nr, dt), picks each solver's best result, and computes
        the Pareto front across solvers.

        Args:
            solvers: List of solver instances
            verbose: Whether to print progress

        Returns:
            CrossSolverAnalysis with per-problem and aggregate results
        """
        analysis = CrossSolverAnalysis(timestamp=datetime.now().isoformat())

        problems = [
            {"alpha": alpha, "ic_type": ic_type}
            for alpha in self.alpha_list
            for ic_type in self.ic_types
        ]

        if verbose:
            mode = "PHYSBO" if self._use_physbo else "grid"
            if self._use_physbo:
                total_probes = (
                    len(problems) * len(solvers) *
                    (self._physbo_n_random + self._physbo_n_bayes)
                )
                print(f"\nCross-solver analysis ({mode}): {len(problems)} problems "
                      f"x {len(solvers)} solvers (~{total_probes} probes)")
            else:
                total_runs = (
                    len(problems) * len(solvers) *
                    len(self.dt_list) * len(self.t_end_list)
                )
                print(f"\nCross-solver analysis ({mode}, nr={self._fixed_nr}): "
                      f"{len(problems)} problems "
                      f"x {len(solvers)} solvers ({total_runs} total runs)")

        for prob in problems:
            alpha = prob["alpha"]
            ic_type = prob["ic_type"]
            key = f"alpha={alpha}_ic={ic_type}"

            front = CrossSolverFront(
                problem_key=key,
                problem=prob,
            )

            if verbose:
                print(f"\n  Problem: {key}")

            for solver in solvers:
                best_point = self._find_best_for_problem(
                    solver, alpha, ic_type
                )
                if best_point is not None:
                    front.points.append(best_point)

                    if verbose:
                        if best_point.is_stable and not np.isnan(best_point.l2_error):
                            print(f"    {solver.name:25s}: L2={best_point.l2_error:.2e}, "
                                  f"time={best_point.wall_time*1000:.2f}ms "
                                  f"(nr={best_point.config['nr']}, dt={best_point.config['dt']})")
                        else:
                            print(f"    {solver.name:25s}: UNSTABLE")

            front.compute()
            analysis.problems[key] = front

            if verbose and front.pareto_optimal:
                pareto_solvers = [p.solver for p in front.pareto_optimal]
                print(f"    Pareto-optimal: {', '.join(pareto_solvers)}")

        analysis.compute_summary()

        if verbose:
            print(f"\n  Overall rankings:")
            for r in analysis.overall_rankings[:5]:
                print(f"    #{analysis.overall_rankings.index(r)+1} {r['solver']:25s}: "
                      f"avg_rank={r['avg_rank']:.1f}, "
                      f"avg_L2={r['avg_error']:.2e}, "
                      f"stability={r['stability_rate']:.0f}%")

            if analysis.coverage_gaps:
                print(f"\n  Coverage gaps ({len(analysis.coverage_gaps)}):")
                for gap in analysis.coverage_gaps:
                    print(f"    {gap['description']}")

        return analysis

    def _find_best_for_problem(
        self,
        solver: SolverBase,
        alpha: float,
        ic_type: str,
    ) -> Optional[ParetoPoint]:
        """Find the best config for a solver on a specific problem.

        Uses PHYSBO if enabled (with fallback to grid sweep on failure),
        otherwise uses grid sweep over (nr, dt, t_end).

        Args:
            solver: Solver instance
            alpha: Nonlinearity parameter
            ic_type: Initial condition type

        Returns:
            Best ParetoPoint or None
        """
        if self._use_physbo:
            try:
                result = self._find_best_for_problem_physbo(solver, alpha, ic_type)
                if result is not None:
                    return result
            except Exception:
                pass  # Fall through to grid sweep

        return self._find_best_for_problem_grid(solver, alpha, ic_type)

    def _find_best_for_problem_grid(
        self,
        solver: SolverBase,
        alpha: float,
        ic_type: str,
    ) -> Optional[ParetoPoint]:
        """Find the best dt via grid sweep with fixed nr.

        Args:
            solver: Solver instance
            alpha: Nonlinearity parameter
            ic_type: Initial condition type

        Returns:
            Best ParetoPoint or None
        """
        best_point = None
        fallback_point = None
        nr = self._fixed_nr

        for dt in self.dt_list:
            for t_end in self.t_end_list:
                config = {
                    "alpha": alpha,
                    "nr": nr,
                    "dt": dt,
                    "t_end": t_end,
                    "ic_type": ic_type,
                }
                point = self._run_single_config(solver, config)

                if fallback_point is None:
                    fallback_point = point

                if point.is_stable and not np.isnan(point.l2_error):
                    if best_point is None or point.l2_error < best_point.l2_error:
                        best_point = point

        return best_point if best_point is not None else fallback_point

    def run_quick_cross_solver(
        self,
        solvers: List[SolverBase],
        verbose: bool = True,
    ) -> CrossSolverAnalysis:
        """Quick cross-solver analysis with reduced parameter space.

        Args:
            solvers: List of solver instances
            verbose: Whether to print progress

        Returns:
            CrossSolverAnalysis
        """
        orig_alpha = self.alpha_list
        orig_dt = self.dt_list

        self.alpha_list = [0.0, 0.5, 1.0]
        self.dt_list = [0.001, 0.0005]

        try:
            return self.analyze_cross_solver(solvers, verbose)
        finally:
            self.alpha_list = orig_alpha
            self.dt_list = orig_dt


# =============================================================================
# Convenience Functions
# =============================================================================

def load_latest_pareto_front(solver_name: str, output_dir: str = None) -> Optional[ParetoFront]:
    """Load the most recent Pareto front for a solver.

    Args:
        solver_name: Name of the solver
        output_dir: Directory containing Pareto front files

    Returns:
        ParetoFront or None if not found
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "pareto_fronts")

    if not os.path.isdir(output_dir):
        return None

    # Find matching files
    files = [
        f for f in os.listdir(output_dir)
        if f.startswith(solver_name) and f.endswith(".json")
    ]

    if not files:
        return None

    # Sort by timestamp (in filename) and get latest
    files.sort(reverse=True)
    path = os.path.join(output_dir, files[0])

    return ParetoFront.load(path)


def load_all_pareto_fronts(output_dir: str = None) -> Dict[str, ParetoFront]:
    """Load latest Pareto front for each solver.

    Args:
        output_dir: Directory containing Pareto front files

    Returns:
        Dict mapping solver name to ParetoFront
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "data", "pareto_fronts")

    if not os.path.isdir(output_dir):
        return {}

    # Find all JSON files
    files = [f for f in os.listdir(output_dir) if f.endswith(".json")]

    # Group by solver name
    solver_files: Dict[str, List[str]] = {}
    for f in files:
        solver_name = f.rsplit("_", 1)[0] if "_" in f else f.replace(".json", "")
        if solver_name not in solver_files:
            solver_files[solver_name] = []
        solver_files[solver_name].append(f)

    # Load latest for each solver
    results = {}
    for solver_name, solver_file_list in solver_files.items():
        solver_file_list.sort(reverse=True)
        path = os.path.join(output_dir, solver_file_list[0])
        try:
            results[solver_name] = ParetoFront.load(path)
        except (json.JSONDecodeError, KeyError):
            pass

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    """Run Pareto analysis on all available solvers."""
    from app.run_benchmark import SOLVERS

    print("=" * 60)
    print("PARETO ANALYSIS")
    print("=" * 60)

    agent = ParetoAnalysisAgent()

    # Cross-solver analysis (main analysis)
    print("\n[1] Cross-Solver Analysis")
    cross = agent.run_quick_cross_solver(SOLVERS, verbose=True)

    # Save cross-solver results
    cross_path = os.path.join(agent.output_dir, "cross_solver_analysis.json")
    cross.save(cross_path)
    print(f"\nCross-solver analysis saved to: {cross_path}")

    # Per-solver analysis
    print("\n[2] Per-Solver Analysis")
    results = agent.run_quick_analysis(SOLVERS, verbose=True)

    print("\n" + "=" * 60)
    print("PER-SOLVER SUMMARY")
    print("=" * 60)

    for solver_name, front in results.items():
        print(f"\n{solver_name}:")
        print(f"  Stability: {front.summary['stability_rate']:.1f}%")
        print(f"  Pareto-optimal: {front.summary['pareto_optimal_count']} points")
        if front.summary.get("min_error") is not None:
            print(f"  Error range: {front.summary['min_error']:.2e} - {front.summary['max_error']:.2e}")
        if front.summary.get("min_time") is not None:
            print(f"  Time range: {front.summary['min_time']*1000:.2f}ms - {front.summary['max_time']*1000:.2f}ms")


if __name__ == "__main__":
    main()

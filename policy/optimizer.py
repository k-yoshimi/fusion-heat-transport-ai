"""Parameter optimizer for solver configuration.

Performs multi-objective optimization to determine optimal (dt, nr)
given solver constraints and target accuracy.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from policy.stability import get_stability, StabilityConstraints


@dataclass
class OptimizationResult:
    """Result of parameter optimization.

    Attributes:
        dt: Recommended time step
        nr: Recommended number of grid points
        estimated_error: Predicted L2 error
        estimated_time: Estimated computation time (relative)
        pareto_rank: Rank in Pareto front (0 = best)
        constraint_satisfied: Whether all stability constraints are met
        notes: Any warnings or notes about the configuration
    """

    dt: float
    nr: int
    estimated_error: float
    estimated_time: float
    pareto_rank: int = 0
    constraint_satisfied: bool = True
    notes: str = ""


class ParameterOptimizer:
    """Multi-objective parameter optimizer for heat transport solvers.

    Optimizes (dt, nr) to minimize both error and computation time
    while respecting solver stability constraints.
    """

    def __init__(
        self,
        nr_candidates: Optional[List[int]] = None,
        dt_candidates: Optional[List[float]] = None,
        time_weight: float = 0.1,
    ):
        """Initialize optimizer.

        Args:
            nr_candidates: List of nr values to consider
            dt_candidates: List of dt values to consider
            time_weight: Weight for time in objective (lambda)
        """
        self.nr_candidates = nr_candidates or [21, 31, 41, 51, 71, 101]
        self.dt_candidates = dt_candidates or [
            0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005
        ]
        self.time_weight = time_weight

    def estimate_max_chi(
        self,
        T0: np.ndarray,
        r: np.ndarray,
        alpha: float,
    ) -> float:
        """Estimate maximum chi from initial profile.

        Args:
            T0: Initial temperature profile
            r: Radial grid
            alpha: Nonlinearity parameter

        Returns:
            Estimated max_chi
        """
        dr = r[1] - r[0]
        dTdr = np.zeros_like(T0)
        dTdr[1:-1] = (T0[2:] - T0[:-2]) / (2 * dr)
        dTdr[0] = (T0[1] - T0[0]) / dr
        dTdr[-1] = (T0[-1] - T0[-2]) / dr

        abs_dTdr = np.abs(dTdr)
        chi = np.full_like(abs_dTdr, 0.1)
        mask = abs_dTdr > 0.5
        chi[mask] = (abs_dTdr[mask] - 0.5) ** alpha + 0.1

        return float(np.max(chi))

    def estimate_error(
        self,
        dt: float,
        nr: int,
        stability: StabilityConstraints,
        t_end: float,
    ) -> float:
        """Estimate L2 error based on discretization parameters.

        Uses error scaling: error ~ O(dt^p + dr^q) where
        p = temporal_order, q = spatial_order.

        Args:
            dt: Time step
            nr: Number of grid points
            stability: Solver stability constraints
            t_end: Final time

        Returns:
            Estimated relative L2 error
        """
        dr = 1.0 / (nr - 1)

        # Temporal error: scales as dt^temporal_order
        p = min(stability.temporal_order, 2)  # Cap at 2 for practical estimates
        temporal_err = (dt / 0.001) ** p * 0.01

        # Spatial error: scales as dr^spatial_order
        q = min(stability.spatial_order, 4)  # Cap at 4 for practical estimates
        spatial_err = (dr / 0.02) ** q * 0.01

        # Time accumulation factor
        time_factor = np.sqrt(t_end / 0.1)

        return (temporal_err + spatial_err) * time_factor

    def estimate_time(
        self,
        dt: float,
        nr: int,
        t_end: float,
        stability: StabilityConstraints,
    ) -> float:
        """Estimate relative computation time.

        Args:
            dt: Time step
            nr: Number of grid points
            t_end: Final time
            stability: Solver stability constraints

        Returns:
            Estimated relative computation time
        """
        nt = int(t_end / dt)

        # Time complexity varies by method
        if stability.spatial_order >= 100:  # Spectral methods
            # O(N^2) or O(N log N) per step
            return nt * nr * np.log(nr + 1)
        else:
            # O(N) for banded solvers
            return nt * nr

    def _is_pareto_dominated(
        self,
        candidate: Tuple[float, float],
        others: List[Tuple[float, float]],
    ) -> bool:
        """Check if candidate is Pareto-dominated by any other point.

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

    def optimize_for_solver(
        self,
        solver_name: str,
        T0: np.ndarray,
        r: np.ndarray,
        alpha: float,
        t_end: float,
        target_error: float = 0.005,
    ) -> OptimizationResult:
        """Find optimal (dt, nr) for a solver.

        Uses multi-objective optimization minimizing error + lambda * time
        subject to stability constraints.

        Args:
            solver_name: Name of the solver
            T0: Initial temperature profile
            r: Radial grid
            alpha: Nonlinearity parameter
            t_end: Final simulation time
            target_error: Target L2 error threshold

        Returns:
            OptimizationResult with recommended parameters
        """
        stability = get_stability(solver_name)
        max_chi = self.estimate_max_chi(T0, r, alpha)

        # Filter nr candidates by solver constraints
        nr_min, nr_max = stability.recommended_nr_range
        nr_list = [
            n for n in self.nr_candidates
            if stability.min_nr <= n <= max(nr_max, max(self.nr_candidates))
        ]
        if not nr_list:
            nr_list = [nr_min]

        # Build candidate configurations
        candidates = []
        for nr in nr_list:
            dr = 1.0 / (nr - 1)
            dt_max = stability.compute_max_dt(dr, max_chi)

            for dt in self.dt_candidates:
                # Skip if violates CFL
                if dt > dt_max:
                    continue

                # Skip if alpha exceeds solver limit
                if not stability.validate_alpha(alpha):
                    continue

                est_error = self.estimate_error(dt, nr, stability, t_end)
                est_time = self.estimate_time(dt, nr, t_end, stability)

                candidates.append({
                    "dt": dt,
                    "nr": nr,
                    "error": est_error,
                    "time": est_time,
                    "objective": est_error + self.time_weight * est_time / 1000,
                })

        if not candidates:
            # Fallback: use minimum stable configuration
            nr = max(stability.min_nr, 31)
            dr = 1.0 / (nr - 1)
            dt = stability.compute_max_dt(dr, max_chi) * 0.8
            if dt == float("inf"):
                dt = 0.001

            return OptimizationResult(
                dt=dt,
                nr=nr,
                estimated_error=0.01,
                estimated_time=1000,
                pareto_rank=0,
                constraint_satisfied=False,
                notes=f"No valid configuration found; using fallback. "
                      f"alpha={alpha} may exceed solver limits.",
            )

        # Compute Pareto ranks
        error_time_pairs = [(c["error"], c["time"]) for c in candidates]
        pareto_ranks = []
        for i, c in enumerate(candidates):
            others = [error_time_pairs[j] for j in range(len(candidates)) if j != i]
            is_dominated = self._is_pareto_dominated(
                (c["error"], c["time"]), others
            )
            pareto_ranks.append(1 if is_dominated else 0)

        for i, rank in enumerate(pareto_ranks):
            candidates[i]["pareto_rank"] = rank

        # Filter to candidates meeting target error
        feasible = [c for c in candidates if c["error"] <= target_error]
        if feasible:
            # Among feasible, pick minimum time
            best = min(feasible, key=lambda c: c["time"])
        else:
            # No feasible; pick Pareto-optimal with best objective
            pareto_optimal = [c for c in candidates if c["pareto_rank"] == 0]
            best = min(pareto_optimal, key=lambda c: c["objective"])

        notes = ""
        if best["error"] > target_error:
            notes = (f"Cannot achieve target_error={target_error:.4g}; "
                     f"estimated error={best['error']:.4g}")

        return OptimizationResult(
            dt=best["dt"],
            nr=best["nr"],
            estimated_error=best["error"],
            estimated_time=best["time"],
            pareto_rank=best["pareto_rank"],
            constraint_satisfied=True,
            notes=notes,
        )

    def optimize_for_accuracy(
        self,
        solver_name: str,
        T0: np.ndarray,
        r: np.ndarray,
        alpha: float,
        t_end: float,
        target_error: float = 0.005,
    ) -> OptimizationResult:
        """Find parameters prioritizing accuracy over speed.

        Args:
            solver_name: Name of the solver
            T0: Initial temperature profile
            r: Radial grid
            alpha: Nonlinearity parameter
            t_end: Final simulation time
            target_error: Target L2 error threshold

        Returns:
            OptimizationResult with accuracy-focused parameters
        """
        # Temporarily reduce time weight
        old_weight = self.time_weight
        self.time_weight = 0.01
        result = self.optimize_for_solver(
            solver_name, T0, r, alpha, t_end, target_error
        )
        self.time_weight = old_weight
        return result

    def optimize_for_speed(
        self,
        solver_name: str,
        T0: np.ndarray,
        r: np.ndarray,
        alpha: float,
        t_end: float,
        max_error: float = 0.02,
    ) -> OptimizationResult:
        """Find fastest parameters within error tolerance.

        Args:
            solver_name: Name of the solver
            T0: Initial temperature profile
            r: Radial grid
            alpha: Nonlinearity parameter
            t_end: Final simulation time
            max_error: Maximum acceptable L2 error

        Returns:
            OptimizationResult with speed-focused parameters
        """
        # Temporarily increase time weight
        old_weight = self.time_weight
        self.time_weight = 1.0
        result = self.optimize_for_solver(
            solver_name, T0, r, alpha, t_end, max_error
        )
        self.time_weight = old_weight
        return result


def optimize_parameters(
    solver_name: str,
    T0: np.ndarray,
    r: np.ndarray,
    alpha: float,
    t_end: float,
    target_error: float = 0.005,
    time_weight: float = 0.1,
) -> OptimizationResult:
    """Convenience function to optimize parameters for a solver.

    Args:
        solver_name: Name of the solver
        T0: Initial temperature profile
        r: Radial grid
        alpha: Nonlinearity parameter
        t_end: Final simulation time
        target_error: Target L2 error threshold
        time_weight: Weight for computation time in objective

    Returns:
        OptimizationResult with recommended parameters
    """
    optimizer = ParameterOptimizer(time_weight=time_weight)
    return optimizer.optimize_for_solver(
        solver_name, T0, r, alpha, t_end, target_error
    )

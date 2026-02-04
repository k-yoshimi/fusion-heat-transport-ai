"""Stability constraints and metadata for heat transport solvers."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class StabilityConstraints:
    """Stability constraints for a numerical solver.

    Attributes:
        is_unconditionally_stable: True for implicit methods (no CFL constraint)
        cfl_coefficient: dt <= cfl * dr^2 / max_chi for explicit methods
        max_alpha: Maximum recommended alpha for stable operation
        spatial_order: Order of spatial accuracy (2, 3, 4, or N for spectral)
        temporal_order: Order of temporal accuracy (1 or 2)
        min_nr: Minimum number of grid points
        recommended_nr_range: (min, max) recommended grid points for balance
    """

    is_unconditionally_stable: bool = False
    cfl_coefficient: Optional[float] = None
    max_alpha: Optional[float] = None
    spatial_order: int = 2
    temporal_order: int = 1
    min_nr: int = 11
    recommended_nr_range: Tuple[int, int] = (31, 101)

    def compute_max_dt(self, dr: float, max_chi: float) -> float:
        """Compute maximum stable time step.

        Args:
            dr: Grid spacing
            max_chi: Maximum diffusivity value

        Returns:
            Maximum stable dt (inf for unconditionally stable methods)
        """
        if self.is_unconditionally_stable:
            return float("inf")
        if self.cfl_coefficient is None:
            return float("inf")
        if max_chi <= 0:
            return float("inf")
        return self.cfl_coefficient * dr * dr / max_chi

    def is_stable(self, dt: float, dr: float, max_chi: float) -> bool:
        """Check if given parameters satisfy stability constraints.

        Args:
            dt: Time step
            dr: Grid spacing
            max_chi: Maximum diffusivity value

        Returns:
            True if stable, False otherwise
        """
        max_dt = self.compute_max_dt(dr, max_chi)
        return dt <= max_dt

    def validate_alpha(self, alpha: float) -> bool:
        """Check if alpha is within recommended range.

        Args:
            alpha: Nonlinearity parameter

        Returns:
            True if alpha is acceptable
        """
        if self.max_alpha is None:
            return True
        return alpha <= self.max_alpha


# Stability metadata registry for all solvers
SOLVER_STABILITY = {
    "implicit_fdm": StabilityConstraints(
        is_unconditionally_stable=True,
        cfl_coefficient=None,
        max_alpha=None,
        spatial_order=2,
        temporal_order=2,  # Crank-Nicolson
        min_nr=11,
        recommended_nr_range=(31, 101),
    ),
    "compact4_fdm": StabilityConstraints(
        is_unconditionally_stable=True,
        cfl_coefficient=None,
        max_alpha=None,
        spatial_order=4,
        temporal_order=2,  # Crank-Nicolson
        min_nr=11,
        recommended_nr_range=(31, 71),
    ),
    "p2_fem": StabilityConstraints(
        is_unconditionally_stable=True,
        cfl_coefficient=None,
        max_alpha=None,
        spatial_order=3,  # Quadratic elements
        temporal_order=2,  # Crank-Nicolson
        min_nr=11,
        recommended_nr_range=(21, 51),
    ),
    "cell_centered_fvm": StabilityConstraints(
        is_unconditionally_stable=True,
        cfl_coefficient=None,
        max_alpha=None,
        spatial_order=2,
        temporal_order=1,  # Backward Euler
        min_nr=11,
        recommended_nr_range=(31, 101),
    ),
    "chebyshev_spectral": StabilityConstraints(
        is_unconditionally_stable=True,
        cfl_coefficient=None,
        max_alpha=None,
        spatial_order=100,  # Spectral (exponential) - use large value
        temporal_order=1,  # Backward Euler
        min_nr=16,
        recommended_nr_range=(16, 64),
    ),
    "spectral_cosine": StabilityConstraints(
        is_unconditionally_stable=False,
        cfl_coefficient=0.5,
        max_alpha=0.5,  # Explicit nonlinear correction limits alpha
        spatial_order=100,  # Spectral
        temporal_order=1,
        min_nr=16,
        recommended_nr_range=(32, 64),
    ),
    "imex_fdm": StabilityConstraints(
        is_unconditionally_stable=False,
        cfl_coefficient=0.4,
        max_alpha=None,  # Adaptive chi_base handles any alpha
        spatial_order=2,
        temporal_order=2,  # Crank-Nicolson for implicit part
        min_nr=11,
        recommended_nr_range=(31, 101),
    ),
    "pinn_stub": StabilityConstraints(
        is_unconditionally_stable=True,  # Not time-stepping
        cfl_coefficient=None,
        max_alpha=None,
        spatial_order=100,  # Neural network approximation
        temporal_order=100,  # Direct solution
        min_nr=11,
        recommended_nr_range=(21, 51),
    ),
}


def get_stability(solver_name: str) -> StabilityConstraints:
    """Get stability constraints for a solver.

    Args:
        solver_name: Name of the solver

    Returns:
        StabilityConstraints for the solver

    Raises:
        KeyError: If solver_name is not in registry
    """
    if solver_name not in SOLVER_STABILITY:
        raise KeyError(f"Unknown solver: {solver_name}. "
                       f"Available: {list(SOLVER_STABILITY.keys())}")
    return SOLVER_STABILITY[solver_name]


def is_solver_stable(
    solver_name: str,
    dt: float,
    nr: int,
    max_chi: float,
    alpha: float = 0.0,
) -> Tuple[bool, str]:
    """Check if a solver configuration is stable.

    Args:
        solver_name: Name of the solver
        dt: Time step
        nr: Number of grid points
        max_chi: Maximum diffusivity value
        alpha: Nonlinearity parameter

    Returns:
        Tuple of (is_stable, message describing any issues)
    """
    try:
        stability = get_stability(solver_name)
    except KeyError as e:
        return False, str(e)

    dr = 1.0 / (nr - 1)
    issues = []

    if nr < stability.min_nr:
        issues.append(f"nr={nr} < min_nr={stability.min_nr}")

    if not stability.validate_alpha(alpha):
        issues.append(f"alpha={alpha} > max_alpha={stability.max_alpha}")

    if not stability.is_stable(dt, dr, max_chi):
        max_dt = stability.compute_max_dt(dr, max_chi)
        issues.append(f"dt={dt} exceeds CFL limit {max_dt:.6g}")

    if issues:
        return False, "; ".join(issues)
    return True, "OK"


def suggest_stable_dt(
    solver_name: str,
    nr: int,
    max_chi: float,
    safety_factor: float = 0.8,
) -> float:
    """Suggest a stable dt for a solver.

    Args:
        solver_name: Name of the solver
        nr: Number of grid points
        max_chi: Maximum diffusivity value
        safety_factor: Multiply CFL limit by this factor (default 0.8)

    Returns:
        Suggested stable dt
    """
    stability = get_stability(solver_name)
    dr = 1.0 / (nr - 1)
    max_dt = stability.compute_max_dt(dr, max_chi)

    if max_dt == float("inf"):
        # For unconditionally stable methods, use accuracy-based heuristic
        # dt ~ O(dr^(temporal_order)) for temporal accuracy
        return 0.001 * safety_factor

    return max_dt * safety_factor
